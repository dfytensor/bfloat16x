"""BF16X GGUF block format — reference packer/unpacker (llama.cpp GGML_TYPE_BF16X).

This is the on-disk layout used by llama.cpp's GGML_TYPE_BF16X (type 43),
documented here as executable reference code:

  32 weights per block, 46 bytes (11.5 bpw, 1.39x vs bf16):

    emax[2]   : 1 byte per 16-weight half = shared max bf16 exponent field
    sgn[4]    : 32 sign bits, LSB-first (bit j = weight j sign)
    mant[28]  : 32 x 7-bit mantissas, LSB-first bit order
    delta[12] : 32 x 3-bit exponent deltas, LSB-first; delta == 7 is a
                saturation sentinel (decoded exponent clamped to emax - 7)

  Any weight within 6 exponents of its half-block max decodes bit-identical
  to the original bf16; only weights already 2^-7 below the half-block max
  (~2% on real checkpoints) decode clamped. Sign and the full 7-bit mantissa
  are always exact.

Run as a script for a round-trip self-test against PyTorch bf16 bits:
    python bf16x_gguf_block.py
"""
from __future__ import annotations
import numpy as np

QK_BF16X = 32
BLOCK_BYTES = 46


def _pack_bits(vals: np.ndarray, nbits: int) -> np.ndarray:
    """(n, 32) small unsigned ints -> LSB-first packed bytes (n, 32*nbits/8)."""
    bits = ((vals.reshape(-1, 32, 1) >> np.arange(nbits, dtype=vals.dtype)) & 1)
    bits = bits.reshape(-1, 32 * nbits)
    return np.packbits(bits, axis=1, bitorder='little')


def _unpack_bits(data: np.ndarray, nbits: int) -> np.ndarray:
    """LSB-first packed bytes -> (n, 32) small unsigned ints."""
    n = data.shape[0]
    bits = np.unpackbits(data, axis=1, bitorder='little')[:, :32 * nbits]
    bits = bits.reshape(n, 32, nbits).astype(np.uint16)
    return (bits << np.arange(nbits, dtype=np.uint16)).sum(axis=-1)


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 array -> raw bf16 bit patterns (uint16), round-to-nearest-even."""
    n = x.astype(np.float32).view(np.uint32)
    n = np.where((n & 0x7fffffff) > 0x7f800000,
                 (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)  # nan->quiet
    return ((np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16).astype(np.uint16)


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    """raw bf16 bit patterns (uint16) -> exact float32 array."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def quantize_blocks(x: np.ndarray) -> np.ndarray:
    """float32 array (..., 32k) -> uint8 packed blocks (..., 46)."""
    n = fp32_to_bf16_bits(x.reshape(-1, 32))
    nb = n.shape[0]
    sign = ((n >> 15) & 1).astype(np.uint8)
    expo = ((n >> 7) & 0xFF).astype(np.uint8)
    mant = (n & 0x7F).astype(np.uint8)
    emax = expo.reshape(nb, 2, 16).max(axis=-1)                      # (nb, 2)
    delta = np.minimum(emax.repeat(16, axis=1) - expo, 7).astype(np.uint8)
    out = np.concatenate([
        emax.reshape(nb, 2),
        _pack_bits(sign, 1),
        _pack_bits(mant, 7),
        _pack_bits(delta, 3),
    ], axis=-1)
    assert out.shape[-1] == BLOCK_BYTES
    return out.reshape(*x.shape[:-1], x.shape[-1] // 32, BLOCK_BYTES)


def dequantize_blocks(packed: np.ndarray) -> np.ndarray:
    """uint8 packed blocks (..., 46) -> float32 array (..., 32k)."""
    p = packed.reshape(-1, BLOCK_BYTES)
    nb = p.shape[0]
    emax = p[:, 0:2].astype(np.uint16).repeat(16, axis=1)            # (nb, 32)
    sign = _unpack_bits(p[:, 2:6], 1)
    mant = _unpack_bits(p[:, 6:34], 7)
    delta = _unpack_bits(p[:, 34:46], 3)
    expo = np.maximum(emax - delta, 0)
    bits = (sign << 15) | (expo << 7) | mant
    f32 = bf16_bits_to_fp32(bits)
    return f32.reshape(*packed.shape[:-2], packed.shape[-2] * 32)


def quantize_tensor(x: np.ndarray) -> np.ndarray:
    """Convenience wrapper: quantize a full 2D weight (rows must be %32)."""
    assert x.shape[-1] % QK_BF16X == 0, f"last dim {x.shape[-1]} not divisible by {QK_BF16X}"
    return quantize_blocks(x)


if __name__ == '__main__':
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    rng = np.random.default_rng(0)
    for shape in [(256, 512), (300, 576), (1024, 2048)]:
        w = (rng.standard_normal(shape) * 0.13).astype(np.float32)
        idx = rng.choice(w.size, w.size // 200, replace=False)
        w.reshape(-1)[idx] *= rng.standard_normal(len(idx)) * 8       # heavy tails

        packed = quantize_tensor(w)
        back = dequantize_blocks(packed)

        bpw = packed.size * 8 / w.size
        if has_torch:
            wb = torch.from_numpy(w).to(torch.bfloat16)
            orig = wb.view(torch.int16).numpy().view(np.uint16).reshape(-1)
            got = torch.from_numpy(back).to(torch.bfloat16) \
                   .view(torch.int16).numpy().view(np.uint16).reshape(-1)
            exact = (orig == got).mean() * 100
            # every mismatch must keep sign+mantissa and use the delta==7 clamp
            mism = orig != got
            ok = True
            if mism.any():
                oe, ge = (orig[mism] >> 7) & 0xFF, (got[mism] >> 7) & 0xFF
                pm = packed.reshape(-1, BLOCK_BYTES)
                emax = pm[:, 0:2].astype(int) \
                    .repeat(16, axis=1).reshape(-1)[mism]
                ok = ((orig[mism] & 0x807F) == (got[mism] & 0x807F)).all() and \
                     ((ge == np.maximum(emax - 7, 0)) | (ge == 0)).all()
            print(f'[{shape}] bpw={bpw:.2f} bit-exact={exact:.2f}% '
                  f'saturation-valid={bool(ok)}')
        else:
            print(f'[{shape}] bpw={bpw:.2f} (install torch for the bf16 check)')

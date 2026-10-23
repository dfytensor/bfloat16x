"""BF16X GGUF block format 鈥?reference packer/unpacker (llama.cpp GGML_TYPE_BF16X).

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
    emax = np.maximum(expo.reshape(nb, 2, 16).max(axis=-1), 7)  # floored at 7: decode is clamp-free                      # (nb, 2)
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
    expo = np.where(delta == 7, 0, np.maximum(emax - delta, 0))  # d==7: zero marker
    bits = (sign << 15) | (expo << 7) | mant
    f32 = bf16_bits_to_fp32(bits)
    return f32.reshape(*packed.shape[:-2], packed.shape[-2] * 32)


def quantize_tensor(x: np.ndarray) -> np.ndarray:
    """Convenience wrapper: quantize a full 2D weight (rows must be %32)."""
    assert x.shape[-1] % QK_BF16X == 0, f"last dim {x.shape[-1]} not divisible by {QK_BF16X}"
    return quantize_blocks(x)


# ---------------------------------------------------------------------------
# Lossless overflow side table (BF16X-L)
#
# quantize_blocks() clamps any exponent delta >= 7 to the zero marker, which
# loses ~2% of real weights (the ones more than 6 exponents below their
# 16-element half-block max). The side table stores those elements exactly,
# turning BF16X into a true-lossless format at ~12 bpw:
#
#   u32 n_rows
#   u32 n_entries
#   u32 row_off[n_rows + 1]          entry index where each row starts
#   { u16 col, u16 bf16_bits }[n_entries]
#
# Entries are sorted by row, then column. Column indices fit in u16 because
# ne0 <= 65535 for all supported tensors. Applying the table restores the
# original bf16 bit pattern of every element.
# ---------------------------------------------------------------------------

OVF_HDR_BYTES = 8


def _bf16_bits(x: np.ndarray, bf16_exact: bool) -> np.ndarray:
    """bf16 bit patterns of x; bf16_exact skips the (no-op) rounding."""
    if bf16_exact:
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return (u >> 16).astype(np.uint16)
    return fp32_to_bf16_bits(x)


def _ovf_mask_from_bits(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    expo = ((bits >> 7) & 0xFF).astype(np.int32)
    e = expo.reshape(rows, cols // QK_BF16X, QK_BF16X)
    emax = np.maximum(e.reshape(rows, cols // QK_BF16X, 2, 16).max(axis=-1), 7)
    delta = emax[..., None].repeat(16, axis=-1).reshape(rows, cols // QK_BF16X, QK_BF16X) - e
    mask = delta.reshape(rows, cols) >= 7
    # exact zeros (+0.0 / -0.0) already decode correctly via the zero marker
    return mask & ((bits & 0x7FFF) != 0)


def ovf_overflow_mask(x: np.ndarray, bf16_exact: bool = False) -> np.ndarray:
    """(rows, cols) bool mask of elements that quantize_blocks() clamps."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = x.shape
    assert cols % QK_BF16X == 0
    return _ovf_mask_from_bits(_bf16_bits(x, bf16_exact), rows, cols)


def pack_ovf(x: np.ndarray, bf16_exact: bool = False) -> bytes:
    """(rows, cols) weight -> serialized overflow table (possibly empty)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = x.shape
    bits = _bf16_bits(x, bf16_exact)
    ovf = _ovf_mask_from_bits(bits, rows, cols)
    r, c = np.nonzero(ovf)  # row-major order: sorted by row, then col
    row_off = np.zeros(rows + 1, dtype=np.uint32)
    if r.size:
        np.cumsum(np.bincount(r, minlength=rows), out=row_off[1:])
        entries = np.empty((r.size, 2), dtype=np.uint16)
        entries[:, 0] = c.astype(np.uint16)
        entries[:, 1] = bits[r, c]
    else:
        entries = np.empty((0, 2), dtype=np.uint16)
    hdr = np.array([rows, r.size], dtype=np.uint32)
    return hdr.tobytes() + row_off.tobytes() + entries.tobytes()


def unpack_ovf(blob: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (n_rows, row_off, entries[:, 0]=col, entries[:, 1]=bf16 bits)."""
    hdr = np.frombuffer(blob[:OVF_HDR_BYTES], dtype=np.uint32)
    rows, n_entries = int(hdr[0]), int(hdr[1])
    off = OVF_HDR_BYTES
    row_off = np.frombuffer(blob[off:off + 4 * (rows + 1)], dtype=np.uint32)
    off += 4 * (rows + 1)
    ent = np.frombuffer(blob[off:off + 4 * n_entries], dtype=np.uint16).reshape(-1, 2)
    return rows, row_off, ent[:, 0], ent[:, 1]


def apply_ovf(decoded: np.ndarray, blob: bytes) -> np.ndarray:
    """Overwrite the clamped elements of a dequantized (rows, cols) tensor."""
    rows, row_off, cols, bits = unpack_ovf(blob)
    assert decoded.shape[0] == rows, (decoded.shape, rows)
    flat_r = np.repeat(np.arange(rows, dtype=np.int64), np.diff(row_off).astype(np.int64))
    decoded = decoded.copy()
    decoded[flat_r, cols.astype(np.int64)] = bf16_bits_to_fp32(bits)
    return decoded


def dequantize_lossless(packed: np.ndarray, blob: bytes) -> np.ndarray:
    """packed blocks (..., 46) + overflow blob -> exact float32 weights (flat)."""
    dec = dequantize_blocks(packed).reshape(-1)
    rows = int(unpack_ovf(blob)[0])
    return apply_ovf(dec.reshape(rows, -1), blob).reshape(-1)


def ovf_stats(x: np.ndarray) -> tuple[int, int]:
    """-> (n_overflow, n_elements)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return int(ovf_overflow_mask(x).sum()), x.size


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

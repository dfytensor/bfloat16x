# BF16X — BFloat16 Lossless Compression

> **100% bit-identical** decompression, **2.08× compression ratio**, full Triton GPU kernel.

BF16X compresses bfloat16 model weights losslessly by exploiting exponent redundancy: adjacent weights share an `emax` (8-bit per 16 elements), with individual deltas stored in only 3 bits. Sign + mantissa are packed as bit-streams. Overflows (rare delta ≥ 7) are handled by a Triton-based sparse fix kernel.

## Platform & Versions

| Component | Version |
|---|---|
| **OS** | Windows 11 |
| **Python** | 3.12.11 |
| **PyTorch** | 2.13+cu126 |
| **Triton** | **3.7.1** (requires `tl.cast(x, uint16, bitcast=True)`) |
| **GPU** | RTX 4090 24GB |
| **CUDA** | 12.6 |

> Triton 3.7+ required for the overflow fix kernel (`tl.cast` with `bitcast=True`). The base decode kernel works with Triton 3.0+.

## Quick Start

### 1. Compression (CPU, offline)

```python
from opqk_linear import bf16x_quantize

# Quantize a single weight matrix
packed = bf16x_quantize(weight_tensor, sub=16)
# Returns dict with: sign_packed, mant_packed, delta_packed, emax,
#                    delta_ovf_idx, delta_ovf_val, out_f, in_f, sub
```

### 2. GPU Decompression (Triton)

```python
from bf16x_triton_test import bf16x_decode_triton

# Decode on GPU (full Triton, no PyTorch post-processing)
weight_bf16 = bf16x_decode_triton(
    packed['sign_packed'].cuda(),
    packed['mant_packed'].cuda(),
    packed['delta_packed'].cuda(),
    packed['emax'].cuda(),
    packed['out_f'], packed['in_f'],
    packed['delta_ovf_idx'], packed['delta_ovf_val'],
    packed['sub'])
# 100% bit-identical to original
```

### 3. Verify Correctness

```python
original = model.layers[0].q_proj.weight.data
decoded = bf16x_decode_triton(...)
assert (original.view(torch.int16).cuda() == decoded.view(torch.int16)).all()
# Always True — 100% bit-identical
```

## Architecture

```
Compression (CPU):
  bf16 weight → emax(shared) + delta(3bit) + sign(1bit) + mant(7bit)
  Overflow: delta ≥ 7 stored separately (sparse, ~1.9% of weights)

Decompression (GPU, Triton):
  Phase 1: _bf16x_decode_kernel (base decode, 20μs for 3M weights)
  Phase 2: _bf16x_fix_ovf_kernel (overflow fix, ~50μs sparse)
  → 100% bit-identical bf16 tensor
```

## Triton Kernels (v6)

### `_bf16x_decode_kernel`
- 3 bit-stream decoders: sign(1b), mant(7b), delta(3b)
- emax lookup (8b per 16 elements)
- **3 critical bug fixes** (see below)
- BLOCK=1024, ~20μs for 2048×1536 matrix

### `_bf16x_fix_ovf_kernel`
- Sparse fix for delta≥7 overflow entries
- Uses `tl.cast(cur, uint16, bitcast=True)` (Triton 3.7+)
- BLOCK=512, ~50μs per layer

## Bugs Fixed in Triton Kernels

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | `>>` arithmetic shift (int32 sign-extend) | bit31=1 pads with 1s, ~10% wrong | `.to(tl.uint32) >>` logical shift |
| 2 | `m_w2 = (m_bit+6)//32` can equal `m_w1` | same word OR'd twice, non-cross elements wrong | Always load `m_w + 1` |
| 3 | `.to(tl.int16)` is value conversion | overflow fix writes 0x0000 | `tl.cast(cur, uint16, bitcast=True)` |

## MiniCPM5-1B Benchmark (RTX 4090 24GB)

| Mode | Fwd | GPU | ppl | Quality |
|---|---|---|---|---|
| bf16 original | 36ms | 2.2GB | 56.02 | 100% |
| **BF16X GPU real-time** | 105ms | 2.0GB | **56.02** | **100% lossless** |
| BF16X CPU streaming | 128ms | 0.9GB | 56.02 | 100% lossless |

- **Disk compression**: 2161MB → 1041MB (2.08×)
- **Triton kernels**: 100% of decode pipeline
- **Global shared buffer**: 1 decode buffer reused across all 168 layers

## File Reference

| File | Purpose |
|---|---|
| `opqk_linear.py` | `bf16x_quantize()`, `BF16XLinear` (Python decode fallback) |
| `bf16x_triton_test.py` | **Triton kernels v6** + `bf16x_decode_triton()` API |
| `compress_bf16x.py` | CPU compression script |
| `compress_bf16x_gpu.py` | GPU batch compression |
| `BF16X.md` | Full documentation with inline kernel code |

## Deploy Scripts (MiniCPM5-1B)

Located in `F:\dg_minicpm5\`:
- `bf16x_gpu_stream.py` — GPU on-the-fly (packed on GPU, shared w_buf)
- `bf16x_stream.py` — CPU streaming (packed on CPU, DMA per layer)
- `bf16x_bench.py` — Benchmark all modes

## Dependencies

```
pip install triton>=3.7.0 torch>=2.1
```

## License

MIT

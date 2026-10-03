# BF16X — BFloat16 Lossless Compression

> **100% bit-identical** decompression, **2.08× compression ratio**, full Triton GPU kernel.
> **v8: fused decode+GEMV — generation speed parity with bf16** (34.8 vs 33-36 ms/tok on MiniCPM5-1B).
> **v9: fused decode+GEMM — multi-token/prefill** (8B image model t2i: 103 s → 56 s/image, 1.84×, 24 GB GPU).

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

| Mode | Gen (ms/tok) | GPU | ppl | Quality |
|---|---|---|---|---|
| bf16 original | 33-36 | 2.2GB | 56.02 | 100% |
| **v8 fused decode+GEMV** | **34.8 (≈bf16)** | **1.98GB** | **56.02** | **100% lossless** |
| v8 opt two-segment | 48.4 | 2.0GB | 56.02 | 100% lossless |
| v6 GPU real-time (per-call) | 353.8* | 2.0GB | 56.02 | 100% lossless |
| CPU streaming | — | 0.9GB | 56.02 | 100% lossless |

\* KV-cache greedy generation, WDDM desktop GPU amplifies the per-layer CPU
sync + DMA cost of the original decode loop.

### v8 fused kernel (`bf16x_fused.py`)

Single-token GEMV with in-register decode — the bf16 weight **never
materializes in VRAM** (reads ~12.3 bpw packed instead of a 32 bpw
decode-write + GEMM-read round trip). The overflow table (delta ≥ 7) becomes
a per-row CSR overlay corrected **inside** the kernel
(`w_corr − w_wrong`·x accumulated directly, no second pass, no CPU sync).
R rows per program (R=4, BK=256 measured best); `out_f % R` auto-fallback.

```python
from bf16x_fused import deploy_bf16x_fused, Bf16xFused

model = ...  # cuda, eval
deploy_bf16x_fused(model)            # replaces all quantizable Linears
# modes per layer: 'fused' (default, single-token) / 'opt' / 'repo';
# multi-token prefill always uses the shared-buffer decode + F.linear
```

Reproduce: `python bench_fused_cpm5.py`

### v9 fused decode+GEMM (`bf16x_gemm_fused.py`)

Multi-token counterpart of the v8 GEMV: W tiles are decoded **in registers**
and fed to `tl.dot` — the bf16 weight never materializes in VRAM, so the hot
path reads ~11.9 bpw packed instead of a 32 bpw decode-write + GEMM-read round
trip. Handles arbitrary `out_f` / `in_f` / `M` (masked edges); decode tile
lives in smem (BN×BK = 128×32 best).

delta==7 saturation applies (near-lossless: true delta>7 hits ~2% of elements;
verified bit-exact vs the saturated decode with one-hot probes). Bit-exact
multi-token: two-segment decode; bit-exact single-token: v8 GEMV.

Measured on SenseNova-U1.5-8B-MoT t2i (588 Linears, 16:9 2720×1536, 10 steps,
RTX 4090 24 GB): **103 s → 56 s per image (1.84×)** vs the two-segment
DMA→decode→F.linear path, with the backbone (294 Linears) GPU-resident
zero-DMA and the fp32-origin branch (294 Linears) pinned-CPU + DMA.

```python
from bf16x_gemm_fused import Bf16xGemmLinear, bf16x_fused_gemm, deploy_bf16x_gemm

deploy_bf16x_gemm(model)   # in-memory: quantize + replace all Linears
# or per-layer:
lay = Bf16xGemmLinear(packed, bias=b, resident=False)   # resident=False -> pinned CPU + DMA
y = model(x)               # prefill / image models / any M > 1
```

### Offline pack to disk (`pack_to_disk.py`)

One-time packing of a whole model into a **single safetensors + meta JSON**,
layer-by-layer on CPU with the original dropped immediately (host-RAM peak ≈
packed streams + one layer: a 50 GB 8B model packs in ~26 GB RAM). Loading
mmap-slices the blob per layer — no re-quantization at startup (62 s for a
588-linear 8B model). The loader provides two-segment **bit-exact** inference
with GPU-resident or pinned-CPU-streamed layers; packed layers can also feed
the fused kernels above.

```python
from pack_to_disk import pack_model_to_disk, replace_linears_from_packed

pack_model_to_disk(model, "model_bf16x")            # once, offline
replace_linears_from_packed(model, "model_bf16x",   # at startup
    resident_fn=lambda name: "mot" not in name)     # GPU-resident vs CPU+DMA
```

Format `bf16x-v1`: `<prefix>.safetensors` holds flat streams per layer
(`<name>.{sign,mant,delta,emax,ovf_i,ovf_v,bias}`), `<prefix>.meta.json` the
per-tensor offsets/shapes. WDDM note: stream pinned pools in ≤256 MB chunks —
repeated multi-GB pins fragment the Windows host allocator.

- **Disk compression**: 2161MB → 1041MB (2.08×)
- **Triton kernels**: 100% of decode pipeline
- **Global shared buffer**: 1 decode buffer reused across all 168 layers

## File Reference

| File | Purpose |
|---|---|
| `opqk_linear.py` | `bf16x_quantize()`, `BF16XLinear` (Python decode fallback) |
| `bf16x_triton_test.py` | **Triton kernels v7** + `bf16x_decode_triton()` API |
| `bf16x_fused.py` | **v8: fused decode+GEMV kernel** + `Bf16xFused` (3 modes) + `deploy_bf16x_fused()` |
| `bf16x_gemm_fused.py` | **v9: fused decode+GEMM kernel** (multi-token) + `Bf16xGemmLinear` + `deploy_bf16x_gemm()` |
| `pack_to_disk.py` | **Offline pack**: whole model → one safetensors + meta; bit-exact streaming loader |
| `bench_fused_cpm5.py` | bf16 vs repo/opt/fused benchmark on MiniCPM5-1B |
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

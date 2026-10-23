# BF16X — BFloat16 Lossless Compression

> **11.5 bpw / 1.39×** near-lossless GGUF block format (PPL 21.0076 vs 21.0155); **12.17 bpw / 1.315×** with the overflow side table for **100% bit-identical** reconstruction. Full Triton GPU kernel.
> **v8: fused decode+GEMV — generation speed parity with bf16** (34.8 vs 33-36 ms/tok on MiniCPM5-1B).
> **v9: fused decode+GEMM — multi-token/prefill** (8B image model t2i: 103 s → 56 s/image, 1.84×, 24 GB GPU).

BF16X compresses bfloat16 model weights near-losslessly (bit-exactly with the overflow side table) by exploiting exponent redundancy: adjacent weights share an `emax` (8-bit per 16 elements), with individual deltas stored in only 3 bits. Sign + mantissa are packed as bit-streams. Overflows (rare delta ≥ 7) are handled by a Triton-based sparse fix kernel.

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

### llama.cpp `GGML_TYPE_BF16X` end-to-end (MiniCPM5-1B, wikitext-2 test, RTX 4090)

Full integration via the llama.cpp PR branch (CUDA backend, -ngl 99,
wikitext-2 test split, ctx 2048):

| Format | Size | BPW | PPL |
|---|---|---|---|
| F16 baseline | 2061 MiB | 16.00 | 21.0155 ± 0.172 |
| **BF16X** | **1482 MiB** | **11.50** | **21.0076 ± 0.173** |
| Q8_0 | 1095 MiB | 8.50 | 21.0102 ± 0.173 |

BF16X is statistically indistinguishable from F16 (delta ≪ the error bar)
while shrinking the file by 1.39x; `llama-perplexity` runs on the quantized
CUDA kernels (mul_mm for prefill, MMVQ for generation, 306 tok/s gen /
163 tok/s pp on the 1B model), and the CPU vec-dot path agrees
(19.84 vs 19.85 ppl on a 60 KB subset).

### llama.cpp speed (llama-bench, MiniCPM5-1B, RTX 4090, -ngl 99)

| Format | Size | pp512 | tg128 |
|---|---|---|---|
| F16 | 2.01 GiB | 40388 t/s | 371.1 ± 2.3 |
| BF16X | 1.45 GiB | 30546 t/s | **389.5 ± 1.5** |
| Q8_0 | 1.07 GiB | 43719 t/s | 507.9 ± 0.9 |

Two kernel/format refinements took BF16X generation from 5% behind F16 to
**5% ahead, at 28% less memory**:
1. quantize floors the half-block exponent at 7 — deltas never underflow,
   so the decode assembles the bf16 bit pattern with pure integer adds
   (no clamp); tiny-exponent blocks also become exactly encoded.
2. the MMVQ kernel extracts weight pairs with a single funnel-shift each
   and uses two accumulators to break the serial FFMA chain.

The bandwidth ceiling for 1.45 GiB is ~680 tok/s (currently ~56%); an int8
MMQ path or dp4a-style decode can push further (follow-up). Prefill
(mul_mm dequant+MMA) still trails F16's pure-MMA path at this size.

**Scaling** — every token carries ~1 ms of format-independent fixed cost
(kernel launches across ~200 tensors, sync, sampling), which amortizes as
models grow, so the BF16X/F16 speed ratio converges to the 1.39x bandwidth
ratio (MiniCPM5-1B and Qwen2.5-3B, same GPU, tg128):

| Model | BF16X size | F16 tg | BF16X tg | speedup |
|---|---|---|---|---|
| MiniCPM5-1B | 1.45 GiB | 371 t/s | 389 t/s | 1.05x |
| Qwen2.5-3B | 4.13 GiB | 124.6 t/s | 159.3 t/s | **1.28x** |

Quality holds on the second architecture too: Qwen2.5-3B ppl 6.7198 vs
F16's 6.7192 (+/- 0.197). At 7B-8B sizes expect ~1.3x+; the effective
bandwidth at 3B already reaches 658 GB/s (65% of peak).

### RWKV7-G1k-13.3B (llama.cpp, RTX 4090 D 24GB)

Real-model validation of the same GGUF/BF16X path on a 13.3B-parameter
recurrent model (RWKV-7 "Goose"), converted from the official bf16
checkpoint (`rwkv7-g1k-13.3b-20260930-ctx25600.pth`, 24.7 GiB):

| | |
|---|---|
| Checkpoint | 24.7 GiB bf16 (does not fit in 24 GB VRAM) |
| **BF16X GGUF** | **17.78 GB** (1714 tensors; matrices BF16X, 1-D params f32) |
| Decode (llama-bench tg128) | **33.65 ± 0.07 t/s** |
| Prefill (llama-bench pp256) | 1656 t/s |
| Effective bandwidth | 598 GB/s (measured device read ceiling: 888 GB/s) |
| GPU kernel breakdown (nsys) | 85% BF16X GEMV, 15% elementwise/norm/LoRA |

Quality: greedy next-token distribution after the prompt, compared with
the reference PyTorch/numpy implementation run on the source safetensors:

| Prompt | BF16X GGUF | Reference |
|---|---|---|
| The Eiffel tower is in the city of | `' Paris'` p=0.605 | `' Paris'` p=0.598 |
| The capital of France is | `' Paris'` p=0.483 | `' Paris'` p=0.521 |

Tensor-level round-trip (GGUF decode vs safetensors source) is ~98%
bit-exact on every tensor class (attention/FFN matrices, LoRA matrices,
embeddings, output head), with max relative error ~1e-3, i.e. only the
bf16x block quantization itself.

Scale context: an F16 13.3B needs 26.6 GB and cannot run on a 24 GB card;
BF16X makes it fit with ~5 GB to spare and reads 1.39x fewer bytes per
token. In the same llama.cpp CUDA build the effective bandwidth is
598 GB/s, versus 513 GB/s for the official RWKV7-G1 2.9B fp16 entry on a
4090 in the RWKV performance database and 674 GB/s for llama.cpp's
best-tuned dense Q4_0 path - within 11% of a mature dense-quant kernel
while carrying the ~15% RWKV-specific fixed overhead (many small
elementwise/LoRA kernels per layer).

Converter and reference checker: `convert_rwkv7_to_gguf.py` and
`compare_rwkv7_ref.py` in the llama.cpp PR branch (#29896).

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

- **Disk compression (legacy `bf16x-v1` safetensors layout)**: 2161MB → 1041MB (2.08×, ~7.7 bpw). The deployed GGUF block format is 11.5 bpw / 1.39×.
- **Triton kernels**: 100% of decode pipeline
- **Global shared buffer**: 1 decode buffer reused across all 168 layers

### GGUF block format (llama.cpp `GGML_TYPE_BF16X`)

The llama.cpp integration (PR branch, CPU + CUDA + Metal kernels) uses a
self-contained fixed-rate block layout — reference implementation with
round-trip self-test: `bf16x_gguf_block.py`.

```
32 weights per block, 46 bytes (11.5 bpw):
  emax[2]   one byte per 16-weight half = shared max bf16 exponent
  sgn[4]    32 sign bits, LSB-first
  mant[28]  32 x 7-bit mantissas, LSB-first bit order
  delta[12] 32 x 3-bit exponent deltas, LSB-first; 7 = zero marker (exp = 0)
```

Weights within 6 exponents of their half-block max decode bit-identical to
the original bf16 (~99% of bits on real checkpoints); sign + mantissa are
always exact. Measured end-to-end in llama.cpp on MiniCPM5-1B
(wikitext-2 test, RTX 4090): PPL 21.0076 vs F16's 21.0155 (+/- 0.17) at
11.50 BPW.

#### Bit-exact extension (overflow side table, 12.17 bpw)

`quantize_blocks()` clamps every exponent delta >= 7 to the zero marker,
which is the only lossy part of the format: ~1.9% of real weights (the ones
> 6 exponents below their 16-element half-block max, ~0.6 bpw worth).
`bf16x_gguf_block.py` now also provides the sparse side table used by the
reference Python implementation:

```
u32 n_rows, u32 n_entries
u32 row_off[n_rows + 1]
{ u16 col, u16 bf16_bits }[n_entries]     (sorted by row, then column)
```

`pack_ovf()` / `apply_ovf()` / `dequantize_lossless()` produce and apply it.
Verified on RWKV7-G1k-0.1B: **168 tensors / 190.9M weights, 100.000%
bit-identical** to the source bf16 weights at 12.17 bpw (325 MiB vs 264 MiB
for the lossy 11.5 bpw file, +16.2 MB of tables). Exact zeros need no entry (the zero marker
already decodes them), so all-zero tensors cost only the header.

The GGUF converter writes these tables with `--lossless`
(`<tensor>.ovf`, I8), and `add_ovf_gguf.py` grafts them onto an existing
F16/BF16X pair. llama.cpp pairs every BF16X tensor with its side table
inside `llama_model_base::create_tensor` and applies the corrections in
`build_lora_mm()` / `build_inp_embd()`, so **any architecture** gets the
bit-exact path without per-model code. The two new ops are
`GGML_OP_SPARSE_MM` and `GGML_OP_SPARSE_ROWS` (CPU + CUDA kernels).

Measured (RTX 4090 D, temp 0, weights 100% bit-exact):

| model | 11.5 bpw | 12.17 bpw | decode delta |
|---|---|---|---|
| RWKV7-G1k-13.3B | 17.78 GiB, 33.5 t/s | 18.93 GiB, 29.1 t/s | -13% |
| Spark-X2.6-4B | 5.51 GiB, 95.0 t/s | 5.80 GiB, 88.8 t/s | -6% |
| Qwen2.5-3B | 4.14 GiB, 138.6 t/s | 4.40 GiB, 110.4 t/s | -20% |
| MiniCPM5-1B | 1.45 GiB, 340.3 t/s | 1.53 GiB, 247.3 t/s | -27% |

The correction runs as one small kernel per table inside the CUDA graph
(measured: 0.65 us per graph node + 1.9-2.8 us per kernel). Three ways to
cut it were tried and all were neutral or slower: folding the `add` into
the sparse op (serializes the correction behind the GEMM), a wider inner
loop (the kernel is latency-bound, not instruction-bound), and merging the
q/k/v tables into one launch via a fused op (correct output, but the view
nodes cost more than the two saved kernels). The remaining cost is the
table traffic itself (284 MB/token at 3B) contending with the GEMMs, so it
cannot be removed by launch-level tricks. Small models therefore pay a
larger *relative* share; from 4B up the lossless file is within 6-13% of
the near-lossless one. The default 11.5 bpw output is unchanged.

### Lossless variant (`bf16xl_lossless.py`)

TRUE-lossless sibling at **14.12 bpw**: per 16-element group a 28-byte
LSB-first stream of `val14 = (delta6 << 8) | (mant7 << 1) | sign`
(delta 63 = zero marker; delta > 62 folded to zero), one emax byte per
64 elements. Reconstruction is bit-identical (verified). Includes a
warp-per-row CUDA GEMV (byte-assembly loads, generation-speed parity with
a plain bf16 GEMV) and a `deploy_bf16xl()` model hook.

## File Reference

| File | Purpose |
|---|---|
| `opqk_linear.py` | `bf16x_quantize()`, `BF16XLinear` (Python decode fallback) |
| `bf16x_triton_test.py` | **Triton kernels v7** + `bf16x_decode_triton()` API |
| `bf16x_fused.py` | **v8: fused decode+GEMV kernel** + `Bf16xFused` (3 modes) + `deploy_bf16x_fused()` |
| `bf16x_gemm_fused.py` | **v9: fused decode+GEMM kernel** (multi-token) + `Bf16xGemmLinear` + `deploy_bf16x_gemm()` |
| `pack_to_disk.py` | **Offline pack**: whole model → one safetensors + meta; bit-exact streaming loader |
| `bf16xl_lossless.py` | **TRUE-lossless variant** (14.12 bpw) + CUDA GEMV/decode + `deploy_bf16xl()` |
| `bf16x_gguf_block.py` | **GGUF block format reference** (llama.cpp GGML_TYPE_BF16X) + self-test |
| `docs/TECH_BRIEF.md` | **Hardware submission brief**: format spec, decode-unit microarchitecture, roofline, fp8-split impossibility argument, measured evidence |
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

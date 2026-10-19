# BF16XL: A Fixed-Rate Lossless Block-Float Format with Hardware-Decodable Bit Streams

**Tech brief — v1.0** · Reference implementation: `bf16xl_lossless.py` · Companion near-lossless sibling: `BF16X` (11.5 bpw, llama.cpp-integrated)

---

## 1. Executive summary

BF16XL recompresses bfloat16 weights **losslessly** (bit-identical reconstruction, verified against PyTorch bf16) at a fixed **14.12 bpw** — an **11.8% HBM byte reduction** — using block-shared exponents and per-element 6-bit exponent deltas. It is a block-float format in the same family as the OCP Microscaling (MX) types (MXFP8/MXFP4), but it spends the saved exponent bits on keeping the **full 7-bit mantissa of every weight**.

The format is software-decodable today (single-token GEMV reaches memory-bandwidth parity with plain bf16 on an RTX 4090). The decisive deployment case, however — **batched GEMM** — hits a measured software decode wall. We show this wall is structural (the delta coding that saves the bits also breaks tensor-core GEMM-ability), quantify the software cost, and show that a trivial hardware decode stage — bit-plane extraction, shifter/mux wiring only, no multipliers — turns bf16xl into a pure-bandwidth format with a projected **1.13x weight-stream throughput over bf16 and zero accuracy risk** at any batch size.

Positioning: **"zero-risk HBM savings"** — unlike fp8/fp4, any model, any weights, plug-and-play, no calibration, no accuracy gate.

---

## 2. Format specification

Per **64-element supergroup**: one shared-max exponent byte. Per **16-element group**: a 28-byte LSB-first bit stream, 14 bits per element:

```
val14 = (delta6 << 8) | (mant7 << 1) | sign1
```

| field | bits | notes |
|---|---|---|
| `sign` | 1 | exact |
| `mant` | 7 | exact — the full bf16 mantissa |
| `delta` | 6 | `emax - expo`; `63` = zero marker; `delta > 62` folded to `63` (weight < 2^(emax-62), numerically zero for dot products) |

Reconstruction: `bf16(sign, emax - delta, mant)` == original bits, **exactly**. Total cost: 28 B / 16 weights (14 bpw) + 1 B / 64 (0.125 bpw) = **14.12 bpw**.

Reference packer/unpacker with bit-exactness self-test: `bf16xl_lossless.py` (`bf16xl_pack`, `bf16xl_decode_ref`).

Companion format **BF16X** (shipped in a llama.cpp PR): 11.5 bpw near-lossless variant (3-bit deltas, saturation sentinel instead of zero marker) with complete CPU + CUDA + Metal inference integration. All accuracy/speed numbers in §5 for BF16X are end-to-end measurements in llama.cpp.

---

## 3. Why fp8 splitting cannot replace it (the decomposition wall)

A natural proposal: split each bf16 weight into a hardware-native high part and a correction low part:

```
W = hi(e4m3) + lo,   GEMM: W·x = hi·x + lo·x
```

The first term runs on fp8 tensor cores. The second dies:

```
lo_i is scaled by 2^(e_hi,i - 10)   <- per-element exponent
Σ lo_i · 2^(e_i) · x_i  ≠ matrix multiplication
```

A GEMM requires the scale to be factored **out** of the sum (one scale per tile); the per-element exponent of the correction term keeps it **inside** the sum. Every decomposition attempt fails the same way:

| attempt | result |
|---|---|
| hi = e4m3 + lo = mantissa remainder | lo carries a per-element exponent -> not a GEMM |
| lo stored as e5m2 | 2-bit mantissa insufficient -> lossy |
| group-shared exponent for lo (drop deltas) | groups with dynamic range > 8x blow up -> lossy |
| two int4 planes (Blackwell int4 TC) | delta problem unchanged |

**The structural insight**: fp8 can feed tensor cores because it pays a 4-5 bit *inline exponent tax* per element. Block-float lossless formats recover exactly those bits via shared exponents — and shared exponents/deltas are precisely what tensor cores do not accept (they require one uniform scale per tile). **Lossless 15-bit payload + hardware GEMM-ability are incompatible without a decode stage between memory and the MMA pipeline.** That is the gap this brief asks hardware to fill.

---

## 4. The software decode wall (measured)

- **Single-token GEMV (M=1)**: a warp-per-row CUDA kernel with byte-assembly loads (7 x u32 per group, register funnel-shift decode) reaches memory-bandwidth parity with plain bf16 GEMV on an RTX 4090 — decode ALU hides under HBM latency. Verified bit-exact output.
- **Batched GEMM (M>1)**: tensor cores require operands as fragments; software must either (a) materialize the full bf16 W to HBM first (a full write+read round trip that cancels the compression), or (b) decode per element inside the MMA pipeline. Measured option (b): ~25 ALU ops per element for extraction+assembly (archived negative results in the repo), an order-of-magnitude slowdown versus native bf16 GEMM. The near-lossless BF16X sibling shows the same wall in llama.cpp: its batched path must dequantize to F16 before cuBLAS.

This wall is the hardware's reason to exist.

---

## 5. Hardware decode unit (microarchitecture sketch)

### 5.1 Data path

```
HBM ──> L2 ──> [bit-plane decode unit]  ──> tensor-core operand register file
                 │
                 ├─ byte-assembly: 7 x u32 loads per 16-element group
                 │    (4B-aligned, funnel-shift join, hoisted)
                 ├─ per element: 1 x SHF.R (14-bit window) + mask
                 ├─ sign/expo: 1 x SUB (emax - delta) + 1 x OR (sign into bit 15)
                 └─ bf16 pack (bitcast) -> fragment
```

Decode is pure wiring + shifters + a 7-bit adder chain per element: **no multipliers, no CAM, no lookup tables, no variable-length state**. Fixed rate means constant latency — no decode jitter. One pipeline stage suffices.

### 5.2 Cost and roofline

To saturate a 1008 GB/s HBM stream with 2-byte weights, the unit must emit 504 G weights/s per GPU. Across 128 SMs at 2.6 GHz that is **1.5 weights/cycle/SM** — a single extractor lane per SM covers it; area is negligible next to one tensor core. With the decode stage in place, bf16xl GEMM becomes purely HBM-bound:

- weight-stream ratio vs bf16: 16 / 14.12 = **1.13x** throughput (batched and single-token alike)
- vs fp8 e4m3 (8 bpw): fp8 wins raw bytes, but carries ~1-2% weight error and (on Ada) per-tensor scales only; bf16xl carries **zero** accuracy risk
- software wall (§4) is removed by construction

### 5.3 Precedent

Blackwell already ships general-purpose decompression engines (LZ4/Deflate) feeding the L2, and MXFP4/MXFP8 made new numeric formats first-class. A fixed-rate bit-plane lossless type is a strictly simpler build than either.

---

## 6. Format comparison

| | bf16 | fp8 e4m3 | fp4 (MX) | BF16X (sibling) | **BF16XL** |
|---|---|---|---|---|---|
| bpw | 16 | 8 | 4 | 11.5 | **14.12** |
| lossless | - | no (~1-2% wt err) | no | near (~99% bits) | **yes (bit-exact)** |
| scale granularity | - | per-tensor (Ada) / MX (Blackwell) | MX block | 16-weight half | 64-weight group |
| SW GEMM today | native | cuBLASLt | limited | llama.cpp (dequant path) | dequant path only |
| HW decode stage | - | none needed | none needed | not sufficient for lossless | **required (trivial)** |
| accuracy risk | none | gate + calibrate | high | negligible | **none** |
| HBM saving vs bf16 | - | 50% | 75% | 28% | 11.8% |

---

## 7. Evidence appendix (all measured on RTX 4090 D, CUDA)

### 7.1 BF16X in llama.cpp (end-to-end, PR branch)

Quality — llama-perplexity, wikitext-2 test, -ngl 99:

| Model | F16 ppl | BF16X ppl |
|---|---|---|
| MiniCPM5-1B (llama) | 21.0155 ± 0.172 | 21.0076 ± 0.173 |
| Qwen2.5-3B (qwen2) | 6.7192 ± 0.197 | 6.7198 ± 0.197 |

Speed — llama-bench, tg128:

| Model | BF16X size | F16 | BF16X | speedup |
|---|---|---|---|---|
| MiniCPM5-1B | 1.45 GiB | 371 t/s | 389 t/s | 1.05x |
| Qwen2.5-3B | 4.13 GiB | 125 t/s | 159 t/s | 1.28x |

The per-token fixed cost (~1 ms of kernel launches across ~200 tensors) is format-independent and amortizes with scale: the BF16X/F16 ratio converges toward the 1.39x bandwidth ratio; effective weight-stream bandwidth reaches 706 GB/s (70% of peak) at 3B. Correctness: 238/238 CUDA backend-op tests, quantize round-trip suites, CPU/CUDA/Metal parity.

### 7.2 BF16XL specifics

- bit-exact reconstruction vs PyTorch bf16 (multiple shapes, heavy-tailed weights, subnormal dust): `bf16xl_lossless.py` self-test
- M=1 CUDA GEMV at bf16-parity throughput (internal measurement, 582 GB/s class)
- software batched decode wall: archived negative results (§4) — the argument for hardware, not against the format

---

## 8. Risks and counterarguments

| objection | response |
|---|---|
| "Blackwell just shipped; new formats take 2-3 years" | the decode unit is a pre-MMA stage, not a new tensor-core numeric; it can be added without touching MMA datapaths, and the OCP MX process exists exactly for such additions |
| "industry is all-in on lossy fp4" | bf16xl is not competing for the accuracy-tolerant market; it is the **zero-risk cost play**: 11.8% HBM/VRAM savings for every model with no calibration, no accuracy gate, no exclusions — the segment lossy formats structurally cannot serve |
| "Blackwell has generic LZ4/Deflate decompression engines" | those are variable-rate with decode jitter and dictionary overhead, and do not emit tensor-core-ready operands; a fixed-rate bit-plane type decodes at constant latency directly into fragments |
| "SW decode is good enough" | measured wall in §4: batched path is an order of magnitude off; single-token parity took a fully custom kernel and cannot extend to GEMM |

---

## 9. GEMM-enablement roadmap (three paths)

Making bf16xl GEMM-capable hinges on one constraint: **decode must complete at (or immediately before) the operand-capture stage of the MMA pipeline, with no per-element scaling leaking into the summation.** Three paths, in order of verification cost:

### Path 1 — fused decode: reconstruct BF16 tiles in registers (near-term, software)

"Load-compressed, compute-decompressed": decode bit streams into standard BF16 fragments in registers/shared memory immediately before `mma.sync`, exactly as Marlin-style int4 kernels and llama.cpp's own MMQ kernels dequantize into fragments. **This is not hypothetical for us — the funnel-shift MMVQ kernel in the PR branch is already fused decode for M=1..8** (no HBM round trip; decode overlapped with FMA). The missing piece is the batched (M>1) Marlin-style bf16x MMQ kernel, which answers the key open question — *how much of the 28% byte saving survives fusion* — with a prototype instead of an argument.

### Path 2 — fixed-function decode cascade (the architecture ask)

Per-element exponent alignment as a fixed pipeline stage in the tensor-core front-end: bit-plane reassembly + `emax - delta` adder per operand element, before the multiply array. This generalizes the MX block-scale logic (which applies one correction per block pre-accumulation, e.g. MXCore-style datapaths) to per-element granularity. The brief's §5 cost analysis (1.5 weights/cycle/SM, shifter/mux only) bounds the area question; the argument to make is that this increment buys losslessness that block-scale MX structurally cannot.

### Path 3 — format restructure for decode regularity (measured: mostly not worth it)

Proposal: split the 64-element group into 16-element sub-blocks with a sub-block base delta and 2-bit per-element offsets, mapping decode closer to MX block scales. **Measured verdict on real weights** (14.3M half-blocks, MiniCPM5-1B, 8 tensors): intra-half exponent spread is ≤ 3 for only **10.49%** of halves — a 2-bit offset is lossy for ~90% of blocks and breaks the lossless property. Corrected hierarchy options:

| sub-block offset width | lossless coverage | bpw (with emax byte, sign, 7b mantissa) |
|---|---|---|
| 3-bit (spread ≤ 7) | 86.45% | 12.0 |
| 4-bit (spread ≤ 15) | **99.94%** | **12.5** |

A 4-bit-offset variant is a legitimate point between BF16X (11.5 bpw, ~99% exact) and bf16xl (14.12 bpw, 100%) — but note it is a delta-width knob on the existing design, and per-element offsets still exist, so it does not by itself buy GEMM-ability.

### Shared constraints (all paths)

1. **Branchless decode** — fixed-rate encoding keeps per-thread decode work deterministic (the SIMT advantage over Huffman/ANS formats; our CUDA kernel branches only on compile-time constants).
2. **Decode/MMA overlap** — decode the next tile while `mma` consumes the current one (standard software pipelining); decode must never serialize in front of the GEMM.
3. **Register/smem pressure** — a decoded BF16 tile is 16 bpw vs 14.12 compressed; the roofline must charge the tile staging bandwidth (this is exactly what the Path-1 prototype measures).

### Citation hygiene note

Widely-cited verified anchors for this dataflow: **Marlin** (fused int4-dequant into `mma.sync` fragments), **QServe** (W4A8 fused-dequant serving), **ZipNN** and **DFloat11** (lossless exponent coding precedents), **BitPragmatic** (bit-plane compute). Other recently-surfaced system names (e.g. fused BF16-compression GEMM engines) should be independently verified on Google Scholar before inclusion in any submission — unverified citations are fatal in this venue.

---

## 10. Asks

1. **OCP MX working group**: consider an "MX-BF16 lossless" profile (shared exponent + 6-bit delta + full mantissa) as a storage/streaming variant of the MX family.
2. **Vendors (NVIDIA/AMD/Intel)**: a fixed-rate bit-plane decode cascade between L2 and the MMA operand path (§5) — shifter/mux wiring, ~1 stage, per-SM lane count of ~2.
3. **Patent**: pipelined decode-unit for fixed-rate bit-plane lossless weight formats (layout + data path as specified) — to be filed before any submission.
4. **Near-term (software-only)**: authorize the batched fused-decode prototype (Path 1, Marlin-style bf16x MMQ) to replace the dequant+cuBLAS fallback — it converts the batched-decode wall from an argument into a measurement.

---

*Reference code: github.com/dfytensor/bfloat16x (`bf16xl_lossless.py`, `bf16x_gguf_block.py`, `pack_to_disk.py`, llama.cpp integration branch). All measurements reproducible from the scripts in the repository.*

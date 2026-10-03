"""BF16X fused decode+GEMM — multi-token (M>1) inference without materializing bf16 W.

Counterpart of bf16x_fused.py (single-token GEMV). W tiles are decoded in
registers and consumed by tl.dot directly: no bf16 weight in VRAM, reads
~11.8 bpw packed instead of a 32 bpw decode-write + GEMM-read round trip.
Measured on SenseNova-U1.5-8B t2i (588 Linears, 16:9, 10 steps, RTX 4090):
103 s -> 56 s per image (1.84x) vs the two-segment DMA->decode->F.linear path.

delta==7 saturation is applied (near-lossless: true delta>7 hits ~1% of
elements). For bit-exact multi-token output use the two-segment decode
(bf16x_triton_test.bf16x_decode_triton); for bit-exact single-token use the
CSR-corrected GEMV (bf16x_fused.bf16x_fused_gemv).

Streams are padded with one zero word at the end (see _pad1) so the
cross-word loads (mant_ptr + mw + 1) never go out of bounds.
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS = True
except Exception:
    _HAS = False


if _HAS:

    @triton.jit
    def _bf16x_gemm_fused_kernel(
        x_ptr, y_ptr,
        sign_ptr, mant_ptr, delta_ptr, emax_ptr,
        M, OUT_F, IN_F,
        GROUP: tl.constexpr,
        K_FULL: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_n = offs_n < OUT_F
        mask_m = offs_m < M

        acc = tl.zeros((BM, BN), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            offs_k = k0 + tl.arange(0, BK)
            pos = (offs_n[:, None] * IN_F + offs_k[None, :]).to(tl.int64)  # [BN, BK]
            # [BN, BK] validity: n-dim always; k-dim only when IN_F % BK != 0
            # (masked x -> 0 contribution; masked emax=0 is harmless)
            if K_FULL:
                vm = mask_n[:, None].broadcast_to((BN, BK))
                xm = mask_m[:, None].broadcast_to((BM, BK))
            else:
                vm = mask_n[:, None] & (offs_k < IN_F)[None, :]
                xm = mask_m[:, None] & (offs_k < IN_F)[None, :]

            sw = tl.load(sign_ptr + pos // 32, mask=vm, other=0).to(tl.uint32)
            sign = ((sw >> (pos % 32)) & 1).to(tl.int32)

            mb = pos * 7
            mw = mb // 32
            ms = (mb % 32).to(tl.int32)
            w1 = tl.load(mant_ptr + mw, mask=vm, other=0).to(tl.uint32)
            cross = (ms + 7) > 32
            w2 = tl.where(cross, tl.load(mant_ptr + mw + 1, mask=vm, other=0).to(tl.uint32),
                          tl.zeros((BN, BK), tl.uint32))
            mant = tl.where(cross, ((w1 >> ms) | (w2 << (32 - ms))) & 0x7F,
                            (w1 >> ms) & 0x7F).to(tl.int32)

            db = pos * 3
            dw = db // 32
            ds = (db % 32).to(tl.int32)
            d1 = tl.load(delta_ptr + dw, mask=vm, other=0).to(tl.uint32)
            cd = (ds + 3) > 32
            d2 = tl.where(cd, tl.load(delta_ptr + dw + 1, mask=vm, other=0).to(tl.uint32),
                          tl.zeros((BN, BK), tl.uint32))
            delta = tl.where(cd, ((d1 >> ds) | (d2 << (32 - ds))) & 0x7,
                             (d1 >> ds) & 0x7).to(tl.int32)

            e8 = tl.load(emax_ptr + pos // GROUP, mask=vm, other=0).to(tl.int32)
            expo = tl.minimum(tl.maximum(e8 - delta, 0), 255)
            bits = ((sign << 15) | (expo << 7) | mant).to(tl.uint16)
            wt = bits.to(tl.bfloat16, bitcast=True)            # [BN, BK]

            xt = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * IN_F + offs_k[None, :],
                         mask=xm, other=0.0)                   # [BM, BK]
            acc += tl.dot(xt, tl.trans(wt))

        tl.store(y_ptr + offs_m[:, None].to(tl.int64) * OUT_F + offs_n[None, :],
                 acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def bf16x_fused_gemm(x, sign, mant, delta, emax, out_f, in_f, group=16,
                     bn=128, bk=32, num_warps=8, num_stages=1):
    """y[M, out_f] = x[M, in_f] @ W.T with W decoded on the fly (bf16 out).

    x: [*, in_f] bf16/fp16/fp32 CUDA; streams: int32 word tensors
    (sign/mant/delta, each padded with one zero word) + uint8 emax.
    Decode tile [BN, BK] lives in smem: keep BN*BK <= ~100KB (128x32 best).
    """
    M = x.numel() // in_f
    x2 = x.reshape(M, in_f)
    BM = 64 if M >= 64 else 16
    y = torch.empty(M, out_f, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(out_f, bn), triton.cdiv(M, BM))
    _bf16x_gemm_fused_kernel[grid](
        x2, y, sign, mant, delta, emax, M, OUT_F=out_f, IN_F=in_f, GROUP=group,
        K_FULL=(in_f % bk == 0), BM=BM, BN=bn, BK=bk,
        num_warps=num_warps, num_stages=num_stages)
    return y


# ------------------------------------------------------------------ #
#  drop-in Linear: resident (GPU) or pinned-CPU + DMA streams
# ------------------------------------------------------------------ #
_CHUNK_MB = 1024          # 256MB if your WDDM host allocator fragments


def _pad1(t):
    return torch.cat([t, torch.zeros(1, dtype=t.dtype)])


class _Pinner:
    """Incremental pinned storage: <=_CHUNK_MB buffers, grows as layers pack."""

    def __init__(self):
        self.buf = None
        self.used = 0
        self.total = 0

    def add(self, t: torch.Tensor) -> torch.Tensor:
        es = t.element_size()
        n = t.numel() * es
        if self.buf is not None:                 # align for the dtype view
            self.used = (self.used + es - 1) // es * es
        if self.buf is None or self.used + n > self.buf.numel():
            sz = max(n + es, _CHUNK_MB << 20)
            self.buf = torch.empty(sz, dtype=torch.uint8).pin_memory()
            self.used = 0
            self.total += sz
        v = self.buf[self.used:self.used + n].view(t.dtype).view(t.shape)
        v.copy_(t)
        self.used += n
        return v


_PIN = _Pinner()


class Bf16xGemmLinear(nn.Module):
    """Multi-token fused decode+GEMM linear (prefill / image models).

    resident=True : packed streams staged to GPU once, zero per-forward DMA
                    (backbone that fits in VRAM).
    resident=False: pinned-CPU streams, DMA per forward (branch that does not
                    fit); the SAME fused kernel consumes them after DMA.
    """

    def __init__(self, packed, bias=None, resident=True, pin=None):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.resident = resident
        streams = {k: _pad1(packed[k]).contiguous()
                   for k in ("sign_packed", "mant_packed", "delta_packed")}
        streams["emax"] = packed["emax"].contiguous()
        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone())
        else:
            self._bias = None
        if resident:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._gpu = {k: v.to(dev) for k, v in streams.items()}
            self._cpu = None
            if self._bias is not None:
                self._bias = self._bias.to(dev)
        else:
            p = pin if pin is not None else _PIN
            self._cpu = {k: p.add(v) for k, v in streams.items()}
            self._gpu = None

    def _streams(self, device):
        if self.resident:
            return self._gpu
        return {k: v.to(device, non_blocking=True) for k, v in self._cpu.items()}

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        g = self._streams(x.device)
        y = bf16x_fused_gemm(
            x, g["sign_packed"], g["mant_packed"], g["delta_packed"], g["emax"],
            self.out_features, self.in_features)
        y = y.view(*x.shape[:-1], self.out_features)
        if self._bias is not None:
            y = y + self._bias.to(x.dtype)
        return y


@torch.no_grad()
def deploy_bf16x_gemm(model, min_numel=1000, skip=("lm_head", "embed"),
                      resident_fn=None, verbose=True):
    """Replace nn.Linears with Bf16xGemmLinear (in-memory quantization path).

    resident_fn(name) -> bool decides GPU-resident vs pinned-CPU streaming;
    default: resident everything (falls back gracefully if VRAM runs out at
    staging time — split with resident_fn for models larger than VRAM).
    """
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.numel() < min_numel:
            continue
        if any(s in name for s in skip):
            continue
        targets.append((name, mod))
    from opqk_linear import bf16x_quantize
    total, n_el = 0, 0
    for name, mod in targets:
        w = mod.weight.data.to("cpu", torch.bfloat16)
        p = bf16x_quantize(w, sub=16)
        bias = mod.bias.data if mod.bias is not None else None
        res = resident_fn(name) if resident_fn else True
        lay = Bf16xGemmLinear(p, bias=bias, resident=res)
        parts = name.split(".")
        parent = model
        for q in parts[:-1]:
            parent = getattr(parent, q)
        setattr(parent, parts[-1], lay)
        total += sum(p[k].numel() * p[k].element_size()
                     for k in ("sign_packed", "mant_packed", "delta_packed", "emax"))
        total += p["delta_ovf_idx"].numel() * 5
        n_el += w.numel()
        mod.weight.data = torch.empty(0)
        del w, p
    if verbose:
        print(f"[bf16x-gemm] {len(targets)} layers | {total/1e6:.0f}MB packed "
              f"({n_el*2/total:.2f}x vs bf16) | bpw={total*8/n_el:.2f}", flush=True)
    return total, n_el

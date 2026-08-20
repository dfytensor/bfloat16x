"""BF16X fused decode+GEMV — single-token inference without materializing bf16 W.

Streams in BF16X are ELEMENT-indexed (sign bit i, mant 7-bit group i, delta
3-bit group i, emax byte i//16) — no bitmaps, no ranks, so a per-row walk is
a pure bit-extraction. The only cross-pass dependency is the overflow table
(delta>=7, ~1.9%): handled in-kernel via a per-row CSR overlay (tpab-style):
the main loop accumulates the saturated (delta=7) value, the overlay adds
(w_correct - w_wrong) * x[k] using the true delta from the table.

Two-segment optimized decode (pre-filtered GPU-resident overflow, no CPU
sync, no per-call DMA) is also provided for comparison — this is the best
possible "decode then F.linear" path, better than the repo's as-is path.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS = True
except Exception:
    _HAS = False


# ------------------------------------------------------------------ #
#  fused GEMV kernel (R rows / program)
# ------------------------------------------------------------------ #
if _HAS:

    @triton.jit
    def _bf16x_gemv_fused_kernel(
        x_ptr, y_ptr,
        sign_ptr, mant_ptr, delta_ptr, emax_ptr,
        olk_ptr, old_ptr, oloff_ptr,     # per-row CSR overflow lists
        IN_F: tl.constexpr,
        OUT_F: tl.constexpr,
        GROUP: tl.constexpr,
        BK: tl.constexpr,
        R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n0 = pid * R
        rows = n0 + tl.arange(0, R)
        base = rows.to(tl.int64) * IN_F
        acc = tl.zeros((R,), tl.float32)

        for k0 in tl.range(0, IN_F, BK):
            kidx = k0 + tl.arange(0, BK)
            offs = base[:, None] + kidx[None, :]

            sw = tl.load(sign_ptr + offs // 32).to(tl.uint32)
            sign = ((sw >> (offs % 32)) & 1).to(tl.int32)

            mb = offs * 7
            mw = mb // 32
            ms = mb % 32
            w1 = tl.load(mant_ptr + mw).to(tl.uint32)
            cross = (ms + 7) > 32
            w2 = tl.where(cross, tl.load(mant_ptr + mw + 1).to(tl.uint32),
                          tl.zeros((R, BK), tl.uint32))
            mant = tl.where(cross, ((w1 >> ms) | (w2 << (32 - ms))) & 0x7F,
                            (w1 >> ms) & 0x7F).to(tl.int32)

            db = offs * 3
            dw = db // 32
            ds = db % 32
            d1 = tl.load(delta_ptr + dw).to(tl.uint32)
            cd = (ds + 3) > 32
            d2 = tl.where(cd, tl.load(delta_ptr + dw + 1).to(tl.uint32),
                          tl.zeros((R, BK), tl.uint32))
            delta = tl.where(cd, ((d1 >> ds) | (d2 << (32 - ds))) & 0x7,
                             (d1 >> ds) & 0x7).to(tl.int32)

            e8 = tl.load(emax_ptr + offs // GROUP).to(tl.int32)
            expo = tl.minimum(tl.maximum(e8 - delta, 0), 255)
            bits = ((sign << 15) | (expo << 7) | mant).to(tl.uint16)
            w = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            x = tl.load(x_ptr + kidx).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        # overflow overlay per row: correct saturated(delta=7) values
        for r in tl.static_range(R):
            n = n0 + r
            lo = tl.load(oloff_ptr + n)
            hi = tl.load(oloff_ptr + n + 1)
            part = 0.0
            for j in tl.range(lo, hi):
                k = tl.load(olk_ptr + j)
                td = tl.load(old_ptr + j).to(tl.int32)
                p = n.to(tl.int64) * IN_F + k
                sw = tl.load(sign_ptr + p // 32).to(tl.uint32)
                sgn = ((sw >> (p % 32)) & 1).to(tl.int32)
                mb = p * 7
                ms = mb % 32
                mw = mb // 32
                m1 = tl.load(mant_ptr + mw).to(tl.uint32)
                m2 = tl.load(mant_ptr + mw + 1).to(tl.uint32)  # stream padded
                mant = ((m1 >> ms) | (m2 << (32 - ms))) & 0x7F
                e = tl.load(emax_ptr + p // GROUP).to(tl.int32)
                e7 = tl.minimum(tl.maximum(e - 7, 0), 255)
                ec = tl.minimum(tl.maximum(e - td, 0), 255)
                bw = ((sgn << 15) | (e7 << 7) | mant).to(tl.uint16)
                bc = ((sgn << 15) | (ec << 7) | mant).to(tl.uint16)
                w_wrong = bw.to(tl.bfloat16, bitcast=True).to(tl.float32)
                w_corr = bc.to(tl.bfloat16, bitcast=True).to(tl.float32)
                part += (w_corr - w_wrong) * tl.load(x_ptr + k).to(tl.float32)
            acc = tl.where(tl.arange(0, R) == r, acc + part, acc)

        tl.store(y_ptr + rows, acc.to(tl.bfloat16))


def bf16x_fused_gemv(x, sign, mant, delta, emax, olk, old, oloff,
                     out_f, in_f, group=16, r=4, bk=256, num_warps=2):
    y = torch.empty(out_f, dtype=torch.bfloat16, device=x.device)
    while r > 1 and out_f % r != 0:
        r //= 2
    assert in_f % bk == 0
    _bf16x_gemv_fused_kernel[(out_f // r,)](
        x.reshape(-1), y,
        sign, mant, delta, emax,
        olk, old, oloff,
        IN_F=in_f, OUT_F=out_f, GROUP=group, BK=bk, R=r,
        num_warps=num_warps,
    )
    return y


# ------------------------------------------------------------------ #
#  deploy: linear module with three forward modes
# ------------------------------------------------------------------ #
_SHARED_DEC = None
_SHARED_DEC_N = 0


def _shared_dec_buf(n, device):
    global _SHARED_DEC, _SHARED_DEC_N
    if _SHARED_DEC is None or _SHARED_DEC_N < n:
        _SHARED_DEC = torch.empty(n, dtype=torch.bfloat16, device=device)
        _SHARED_DEC_N = n
    return _SHARED_DEC


def _pad1(t):
    return torch.cat([t, torch.zeros(1, dtype=t.dtype)])


class Bf16xFused(nn.Module):
    """mode: 'repo' (original per-call path), 'opt' (GPU-resident two-segment),
    'fused' (single kernel, W never materialized)."""
    R, BK, WARPS = 4, 256, 2

    def __init__(self, packed, bias=None, keep_cpu=True):
        super().__init__()
        self.out_features = packed["out_f"]
        self.in_features = packed["in_f"]
        self.N = self.out_features * self.in_features
        self.mode = "fused"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._sign = _pad1(packed["sign_packed"]).to(dev)
        self._mant = _pad1(packed["mant_packed"]).to(dev)
        self._delta = _pad1(packed["delta_packed"]).to(dev)
        self._emax = packed["emax"].to(dev)
        # pre-filtered overflow (true delta > 7 only), GPU-resident CSR
        idx = packed["delta_ovf_idx"].to(torch.int64)
        val = packed["delta_ovf_val"].to(torch.int64)
        keep = val > 7
        idx, val = idx[keep], val[keep]
        rows = idx // self.in_features
        k = idx % self.in_features
        order = torch.argsort(rows, stable=True)
        rows, k, val = rows[order], k[order], val[order]
        counts = torch.bincount(rows, minlength=self.out_features)
        offs = torch.zeros(self.out_features + 1, dtype=torch.int32)
        offs[1:] = counts.cumsum(0).to(torch.int32)
        self._olk = k.to(torch.int32).to(dev)
        self._old = val.to(torch.int32).to(dev)
        self._oloff = offs.to(dev)
        self._n_fix = int(idx.numel())
        self._pos64 = idx.to(dev)          # for the scatter fix kernel
        self._val32 = val.to(torch.int32).to(dev)
        if keep_cpu:                        # for 'repo' mode (as-is behavior)
            self._cpu = {kk: packed[kk] for kk in
                         ("sign_packed", "mant_packed", "delta_packed", "emax",
                          "delta_ovf_idx", "delta_ovf_val")}
        else:
            self._cpu = None
        if bias is not None:
            self.register_buffer("_bias", bias.detach().clone().to(dev))
        else:
            self._bias = None

    def _decode_opt(self):
        from bf16x_triton_test import _bf16x_decode_kernel, _bf16x_fix_ovf_kernel
        buf = _shared_dec_buf(self.N, self._sign.device)[: self.N]
        grid = (triton.cdiv(self.N, 1024),)
        _bf16x_decode_kernel[grid](
            self._sign, self._mant, self._delta, self._emax,
            buf, self.N, BLOCK=1024)
        if self._n_fix > 0:
            _bf16x_fix_ovf_kernel[(triton.cdiv(self._n_fix, 512),)](
                buf, self._emax, self._pos64, self._val32,
                self._n_fix, SUB=16, BLOCK=512)
        return buf.view(self.out_features, self.in_features)

    def forward(self, x):
        if self.mode == "fused" and x.numel() == self.in_features and x.is_cuda:
            y = bf16x_fused_gemv(
                x, self._sign, self._mant, self._delta, self._emax,
                self._olk, self._old, self._oloff,
                self.out_features, self.in_features,
                r=Bf16xFused.R, bk=Bf16xFused.BK,
                num_warps=Bf16xFused.WARPS)
            if self._bias is not None:
                y = y + self._bias.to(x.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        if self.mode == "repo" and self._cpu is not None and x.is_cuda:
            from bf16x_triton_test import bf16x_decode_triton
            # streams GPU-resident (their README usage), overflow table stays
            # CPU — reproduces the per-layer [fixable].cuda() DMA + .item()
            # sync that dominates their as-is decode loop
            w = bf16x_decode_triton(
                self._sign, self._mant, self._delta, self._emax,
                self.out_features, self.in_features,
                self._cpu["delta_ovf_idx"], self._cpu["delta_ovf_val"], 16)
            return F.linear(x, w, self._bias)
        w = self._decode_opt()
        return F.linear(x, w, self._bias)


@torch.no_grad()
def deploy_bf16x_fused(model, verbose=True):
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.numel() < 1000:
            continue
        if any(s in name for s in ("lm_head", "embed")):
            continue
        targets.append((name, mod))
    from opqk_linear import bf16x_quantize
    total = 0
    n_el = 0
    for name, mod in targets:
        w = mod.weight.data.to("cpu", torch.bfloat16)
        p = bf16x_quantize(w, sub=16)
        bias = mod.bias.data if mod.bias is not None else None
        lay = Bf16xFused(p, bias=bias)
        parts = name.split(".")
        parent = model
        for q in parts[:-1]:
            parent = getattr(parent, q)
        setattr(parent, parts[-1], lay)
        total += p["sign_packed"].numel() * 4 + p["mant_packed"].numel() * 4 \
            + p["delta_packed"].numel() * 4 + p["emax"].numel() \
            + p["delta_ovf_idx"].numel() * 5
        n_el += w.numel()
        mod.weight.data = torch.empty(0)
        del w, p
    if verbose:
        print(f"[bf16x-fused] {len(targets)} layers | {total/1e6:.0f}MB packed "
              f"({n_el*2/total:.2f}x) | bpw={total*8/n_el:.2f}", flush=True)
    return total, n_el

"""One-time BF16X packing of a model -> single safetensors + meta JSON.

Loads/packs layer-by-layer on CPU and drops each original weight right after
packing, so host-RAM peak ~= packed streams + one bf16 layer (e.g. an 8B/50GB
model packs in ~26GB of RAM). The loader mmaps and slices the blob per layer —
no re-quantization at startup (62s for a 588-linear 8B model vs minutes of
in-memory quantization), and the same file feeds both the two-segment
(bit-exact) and fused-GEMM inference paths.

On-disk format "bf16x-v1" (stable):
  <out>.safetensors : flat streams per layer
                      <name>.sign / .mant / .delta  (int32 bit-stream words,
                      padded with one zero word so cross-word reads are safe)
                      <name>.emax                   (uint8, one per 16 elems)
                      <name>.ovf_i / .ovf_v         (true delta>7 table)
  <out>.meta.json   : {"kind": "bf16x-v1", "tensors": {name: {
                        sign, mant, delta, emax, ovf_i, ovf_v,
                        out_f, in_f, N, n_fix}}}

Usage:
    from pack_to_disk import pack_model_to_disk, replace_linears_from_packed
    pack_model_to_disk(model, "model_bf16x")          # once, offline
    replace_linears_from_packed(model, "model_bf16x") # at startup
"""
from __future__ import annotations
import json
import os
import time
import gc
import torch
import torch.nn as nn

_DEFAULT_SKIP = ("embed", "lm_head")


def _pad1(t):
    return torch.cat([t, torch.zeros(1, dtype=t.dtype)])


@torch.no_grad()
def pack_model_to_disk(model, out_prefix, min_numel=4096, sub=16,
                       skip=_DEFAULT_SKIP, verbose=True):
    """Pack every nn.Linear (numel >= min_numel, name not containing `skip`)
    into <out_prefix>.safetensors + <out_prefix>.meta.json.

    Non-packed small/Linears and other params are left untouched in the model
    (save them yourself, e.g. model.save_pretrained).
    Returns (n_packed, packed_bytes).
    """
    from opqk_linear import bf16x_quantize
    from safetensors.torch import save_file

    blobs = {}
    meta = {"kind": "bf16x-v1", "tensors": {}}
    total, n = 0, 0
    t0 = time.time()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < min_numel or any(s in name for s in skip):
            continue
        w = mod.weight.data.detach().to(torch.bfloat16).cpu()
        p = bf16x_quantize(w, sub=sub)
        oi = p["delta_ovf_idx"].to(torch.int64)
        ov = p["delta_ovf_val"].to(torch.int64)
        keep = ov > 7                       # sentinel 7 stays saturated
        oi, ov = oi[keep], ov[keep]
        m = {}
        for key, val in (("sign", _pad1(p["sign_packed"])),
                         ("mant", _pad1(p["mant_packed"])),
                         ("delta", _pad1(p["delta_packed"])),
                         ("emax", p["emax"]),
                         ("ovf_i", oi), ("ovf_v", ov.to(torch.int32))):
            gk = f"{name}.{key}"
            blobs[gk] = val.contiguous()
            total += val.numel() * val.element_size()
            m[key] = gk
        m["out_f"], m["in_f"] = mod.out_features, mod.in_features
        m["N"] = int(mod.weight.numel())
        m["n_fix"] = int(oi.numel())
        if mod.bias is not None:
            gk = f"{name}.bias"
            blobs[gk] = mod.bias.data.detach().to(torch.bfloat16).cpu().contiguous()
            total += blobs[gk].numel() * blobs[gk].element_size()
            m["bias"] = gk
            mod.bias.data = torch.empty(0)
        meta["tensors"][name] = m
        mod.weight.data = torch.empty(0)    # drop the original now
        del w, p
        n += 1
        if verbose and n % 50 == 0:
            print(f"  {n} packed, blob {total/1e9:.1f}GB, {time.time()-t0:.0f}s",
                  flush=True)
        gc.collect()

    pack_path = out_prefix + ".safetensors"
    meta_path = out_prefix + ".meta.json"
    if verbose:
        print(f"saving {n} packed linears ({total/1e9:.2f}GB) ...", flush=True)
    save_file(blobs, pack_path, metadata={"format": "bf16x"})
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    if verbose:
        print(f"  -> {pack_path} ({os.path.getsize(pack_path)/1e9:.2f}GB)", flush=True)
    return n, total


class Bf16xStreamLinear(nn.Module):
    """Two-segment bit-exact BF16X linear fed from the on-disk blob.

    resident=True : streams staged to GPU once (zero per-forward DMA) — use
                    for the part of the model that fits in VRAM.
    resident=False: pinned-CPU streams + DMA per forward.
    Decode uses the v7 Triton kernels from bf16x_triton_test and a shared
    bf16 buffer, then cublas F.linear.
    """

    _DEC = None
    _DEC_N = 0
    _CHUNK_MB = 256        # small chunks: repeated multi-GB pins fragment the
                           # WDDM host allocator (driver-level, not RAM)

    def __init__(self):
        super().__init__()
        raise RuntimeError("use from_packed()")

    @classmethod
    def from_packed(cls, entry, blob, resident=True):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.out_features = entry["out_f"]
        obj.in_features = entry["in_f"]
        obj.N = entry["N"]
        obj.n_fix = entry.get("n_fix", 0)
        obj.resident = resident
        keys = ("sign", "mant", "delta", "emax")
        if resident:
            obj.gpu = {k: blob[k].cuda() for k in keys}
            if obj.n_fix:
                obj.gpu["pos"] = blob["ovf_i"].cuda()
                obj.gpu["val"] = blob["ovf_v"].cuda()
            obj.pin = None
        else:
            pinner = cls._pinner()
            obj.pin = {k: pinner.add(blob[k]) for k in keys}
            if obj.n_fix:
                obj.pin["pos"] = pinner.add(blob["ovf_i"])
                obj.pin["val"] = pinner.add(blob["ovf_v"])
            else:
                obj.pin["pos"] = torch.zeros(0, dtype=torch.int64)
                obj.pin["val"] = torch.zeros(0, dtype=torch.int32)
        if entry.get("bias"):
            obj.bias = blob["bias"]
        else:
            obj.bias = None
        return obj

    _PIN = None

    @classmethod
    def _pinner(cls):
        if cls._PIN is None:
            class _P:
                def __init__(self):
                    self.buf = None
                    self.used = 0
                    self.total = 0

                def add(self, t):
                    es = t.element_size()
                    n = t.numel() * es
                    if self.buf is not None:
                        self.used = (self.used + es - 1) // es * es
                    if self.buf is None or self.used + n > self.buf.numel():
                        sz = max(n + es, cls._CHUNK_MB << 20)
                        self.buf = torch.empty(sz, dtype=torch.uint8).pin_memory()
                        self.used = 0
                        self.total += sz
                    v = self.buf[self.used:self.used + n].view(t.dtype).view(t.shape)
                    v.copy_(t)
                    self.used += n
                    return v
            cls._PIN = _P()
        return cls._PIN

    def _decode(self, device):
        from bf16x_triton_test import _bf16x_decode_kernel, _bf16x_fix_ovf_kernel
        import triton
        g = self.gpu if self.resident else \
            {k: self.pin[k].to(device, non_blocking=True) for k in self.pin}
        if Bf16xStreamLinear._DEC is None or Bf16xStreamLinear._DEC_N < self.N:
            Bf16xStreamLinear._DEC = torch.empty(
                self.N, dtype=torch.bfloat16, device=device)
            Bf16xStreamLinear._DEC_N = self.N
        buf = Bf16xStreamLinear._DEC[:self.N]
        _bf16x_decode_kernel[(triton.cdiv(self.N, 1024),)](
            g["sign"], g["mant"], g["delta"], g["emax"], buf,
            self.N, BLOCK=1024)
        if self.n_fix:
            _bf16x_fix_ovf_kernel[(triton.cdiv(self.n_fix, 512),)](
                buf, g["emax"], g["pos"], g["val"], self.n_fix, SUB=16, BLOCK=512)
        return buf.view(self.out_features, self.in_features)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        w = self._decode(x.device)
        b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)


@torch.no_grad()
def replace_linears_from_packed(model, out_prefix, resident_fn=None,
                                keep_rest_bf16=True, verbose=True):
    """Swap every packed nn.Linear for Bf16xStreamLinear (mmap-sliced, no
    requantization). resident_fn(name) -> bool: GPU-resident vs CPU-streamed
    (default: resident everything). Remaining Linears stay bf16 when
    keep_rest_bf16. Returns n_replaced."""
    from safetensors import safe_open

    pack_path = out_prefix + ".safetensors"
    meta_path = out_prefix + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)["tensors"]
    n = n_res = n_dma = 0
    t0 = time.time()
    with safe_open(pack_path, framework="pt", device="cpu") as sf:
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if name in meta:
                entry = meta[name]
                blob = {k: sf.get_tensor(v) for k, v in
                        (("sign", entry["sign"]), ("mant", entry["mant"]),
                         ("delta", entry["delta"]), ("emax", entry["emax"]),
                         ("ovf_i", entry["ovf_i"]), ("ovf_v", entry["ovf_v"]),
                         ("bias", entry.get("bias")))
                        if entry.get(k)}
                res = resident_fn(name) if resident_fn else True
                new = Bf16xStreamLinear.from_packed(entry, blob, resident=res)
                parent = model
                parts = name.split(".")
                for q in parts[:-1]:
                    parent = getattr(parent, q)
                setattr(parent, parts[-1], new)
                mod.weight.data = torch.empty(0)
                if res:
                    n_res += 1
                else:
                    n_dma += 1
                n += 1
                del blob
                if n % 100 == 0:
                    gc.collect()
            elif keep_rest_bf16:
                mod.weight.data = mod.weight.data.to(torch.bfloat16)
    if torch.cuda.is_available():
        model.cuda()
        torch.cuda.empty_cache()
    if verbose:
        pin_total = Bf16xStreamLinear._PIN.total if Bf16xStreamLinear._PIN else 0
        print(f"[bf16x] {n} linears from disk in {time.time()-t0:.0f}s | "
              f"{n_res} resident + {n_dma} DMA | pinned {pin_total/1e9:.1f}GB",
              flush=True)
    return n

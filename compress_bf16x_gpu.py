"""GPU BF16X 压缩."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, os, sys, time, gc, glob
import safetensors, safetensors.torch
sys.path.insert(0, r'E:\minimax_h3_run')
from opqk_linear import bf16x_quantize

SRC = r"F:\MiniMax-H3\FL2VA\transformer"
OUT = r"F:\MiniMax-H3-NF4\transformer_bf16x.safetensors"
shards = sorted(glob.glob(SRC + r"\model-*.safetensors"))

EXCLUDE = ["video_patch_proj.", "audio_patch_proj.", "condition_proj.",
           "time_embedder.", "final_layer.audio_out.", "final_layer.video_out."]

def q(key, val):
    return (key.endswith(".weight") and val.dim() == 2 and val.numel() >= 4096
            and not any(key.startswith(ep) for ep in EXCLUDE))

def chunked_quantize(val, chunk_rows=200):
    """分块 BF16X 压缩, 超大层专用."""
    import torch as _t
    out_f, in_f = val.shape
    sp_l = []; mp_l = []; dp_l = []; em_l = []; oi_l = []; ov_l = []
    offset = 0
    for r0 in range(0, out_f, chunk_rows):
        r1 = min(r0 + chunk_rows, out_f)
        b = bf16x_quantize(val[r0:r1].to(_t.bfloat16))
        sp_l.append(b['sign_packed']); mp_l.append(b['mant_packed'])
        dp_l.append(b['delta_packed']); em_l.append(b['emax'])
        if b['delta_ovf_idx'].numel() > 0:
            oi_l.append(b['delta_ovf_idx'] + offset); ov_l.append(b['delta_ovf_val'])
        offset += (r1 - r0) * in_f
        _t.cuda.empty_cache()
    return dict(sign_packed=_t.cat(sp_l), mant_packed=_t.cat(mp_l), delta_packed=_t.cat(dp_l),
                emax=_t.cat(em_l),
                delta_ovf_idx=_t.cat(oi_l) if oi_l else _t.zeros(0, dtype=_t.int32),
                delta_ovf_val=_t.cat(ov_l) if ov_l else _t.zeros(0, dtype=_t.uint8),
                out_f=out_f, in_f=in_f, sub=16)

print(f"GPU compress {len(shards)} shards → {OUT}", flush=True)
t0 = time.time(); all_out = {}; n = 0

for si, shard in enumerate(shards):
    sys.stdout.write(f"  {os.path.basename(shard)} "); sys.stdout.flush()
    data = safetensors.torch.load_file(shard)
    for key, val in data.items():
        if q(key, val):
            nel = val.numel()
            if nel > 100_000_000:
                b = chunked_quantize(val)
            elif nel > 50_000_000:
                b = chunked_quantize(val)
            elif nel > 2_000_000:
                b = bf16x_quantize(val.cuda().to(torch.bfloat16))
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            else:
                b = bf16x_quantize(val.to(torch.bfloat16))
            for bk in ('emax','sign_packed','mant_packed','delta_packed','delta_ovf_idx','delta_ovf_val'):
                all_out[f"{key}.bf16x.{bk}"] = b[bk].cpu()
            all_out[f"{key}.bf16x.out_f"] = torch.tensor(b['out_f'])
            all_out[f"{key}.bf16x.in_f"] = torch.tensor(b['in_f'])
            n += 1
        else:
            all_out[key] = val
    del data; gc.collect()
    torch.cuda.empty_cache()
    sys.stdout.write(f"({n} l)\n"); sys.stdout.flush()

    # 每 6 个 shard 存盘释放内存
    if (si + 1) % 6 == 0 or si == len(shards) - 1:
        part_out = OUT.replace('.safetensors', f'_p{si//6}.safetensors')
        safetensors.torch.save_file(all_out, part_out, metadata={"format":"pt"})
        print(f"    saved {part_out} ({sum(v.numel()*v.element_size() for v in all_out.values())/1e9:.1f}GB)", flush=True)
        all_out.clear(); gc.collect()

# 合并分片
print("merging...", flush=True)
merged = {}
for part in sorted(glob.glob(OUT.replace('.safetensors', '_p*.safetensors'))):
    d = safetensors.torch.load_file(part)
    merged.update(d)
    del d
safetensors.torch.save_file(merged, OUT, metadata={"format":"pt","compression":"bf16x"})
for part in glob.glob(OUT.replace('.safetensors', '_p*.safetensors')):
    os.remove(part)
print(f"done {time.time()-t0:.0f}s, {os.path.getsize(OUT)/1e9:.1f}GB", flush=True)

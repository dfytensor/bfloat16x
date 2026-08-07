"""BF16X 无损压缩 FL2VA transformer → safetensors."""
import os, sys, time, torch, gc, glob
import safetensors, safetensors.torch
sys.path.insert(0, r'E:\minimax_h3_run')
from opqk_linear import bf16x_quantize

SRC = r"F:\MiniMax-H3\FL2VA\transformer"
OUT = r"F:\MiniMax-H3-NF4\transformer_bf16x.safetensors"
shards = sorted(glob.glob(SRC + r"\model-*.safetensors"))

EXCLUDE = ["video_patch_proj.", "audio_patch_proj.", "condition_proj.",
           "time_embedder.", "final_layer.audio_out.", "final_layer.video_out."]

def should_compress(key, val):
    if not key.endswith(".weight"): return False
    if val.dim() != 2: return False
    if val.numel() < 4096: return False
    for ep in EXCLUDE:
        if key.startswith(ep): return False
    return True

print(f"=== BF16X compressing {len(shards)} shards → {OUT} ===", flush=True)
t0 = time.time()
all_out = {}
n_comp = 0; n_raw = 0
total_orig = 0; total_comp = 0

for shard in shards:
    fn = os.path.basename(shard)
    sys.stdout.write(f"  {fn}... "); sys.stdout.flush()
    data = safetensors.torch.load_file(shard)

    for key, val in data.items():
        total_orig += val.numel() * val.element_size()
        if should_compress(key, val):
            b = bf16x_quantize(val)
            # 存 BF16X buffers
            for bk in ('emax', 'sign_packed', 'mant_packed', 'delta_packed',
                       'delta_ovf_idx', 'delta_ovf_val'):
                all_out[f"{key}.bf16x.{bk}"] = b[bk].contiguous()
            all_out[f"{key}.bf16x.out_f"] = torch.tensor(b['out_f'], dtype=torch.int32)
            all_out[f"{key}.bf16x.in_f"] = torch.tensor(b['in_f'], dtype=torch.int32)
            total_comp += sum(v.numel() * v.element_size() for v in b.values() if isinstance(v, torch.Tensor))
            n_comp += 1
        else:
            all_out[key] = val.contiguous()
            total_comp += val.numel() * val.element_size()
            n_raw += 1
    del data; gc.collect()
    print("ok", flush=True)

print(f"\n  compressed: {n_comp} layers, raw: {n_raw}")
print(f"  bf16: {total_orig/1e9:.1f}GB → BF16X: {total_comp/1e9:.1f}GB  ({total_orig/max(total_comp,1):.2f}x)")
safetensors.torch.save_file(all_out, OUT, metadata={"format": "pt", "compression": "bf16x"})
print(f"  saved: {os.path.getsize(OUT)/1e9:.1f} GB, {time.time()-t0:.0f}s", flush=True)

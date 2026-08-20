"""MiniCPM5-1B: bf16 vs BF16X (repo-as-is / optimized two-segment / fused GEMV).

Measures prefill ms and KV-cache generation ms/tok on 4 prompts; checks
greedy-token agreement with bf16 (BF16X is lossless -> should match).
"""
import sys, time, gc
sys.setrecursionlimit(10000)
import pandas  # must precede transformers (env stack-overflow fix)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = r"F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots\4e9de7a0778dc1c362e983e6858f0e77542cbdca"

PROMPTS = [
    "The theory of relativity states that",
    "人工智能的发展历史可以概括为",
    "def quick_sort(arr):",
    "北京最值得游览的三个景点是",
]

tok = AutoTokenizer.from_pretrained(MD)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


@torch.no_grad()
def gen(m, prompt, n=60):
    ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
    out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [nxt[0, 0].item()]
    for _ in range(3):
        out = m(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n - 1):
        out = m(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt[0, 0].item())
    torch.cuda.synchronize()
    return toks, (time.time() - t0) / (n - 1) * 1000


@torch.no_grad()
def prefill_ms(m, n=10):
    ids = tok("Hello " * 20, return_tensors="pt")["input_ids"].cuda()
    m.eval()
    for _ in range(3):
        m(ids)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        m(ids)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


# ---------------- bf16 baseline ----------------
print("== bf16 ==", flush=True)
m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16).cuda().eval()
pf = prefill_ms(m)
ref = []
for p in PROMPTS:
    t, ms = gen(m, p)
    ref.append(t)
    print(f"  {ms:.1f} ms/tok | {p[:40]!r}", flush=True)
print(f"  prefill={pf:.0f}ms gpu={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
del m; gc.collect(); torch.cuda.empty_cache()

# ---------------- BF16X deploy once, 3 modes ----------------
print("== deploy BF16X (quantize 168 layers, CPU) ==", flush=True)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16).cuda().eval()
from bf16x_fused import deploy_bf16x_fused, Bf16xFused
deploy_bf16x_fused(m)
m.eval()
print(f"  deploy took {time.time()-t0:.0f}s", flush=True)

pf = prefill_ms(m)  # opt decode path for prefill
print(f"  prefill(opt-decode)={pf:.0f}ms", flush=True)

def run_mode(mode, sweep=None):
    outs = []
    for lay in m.modules():
        if isinstance(lay, Bf16xFused):
            lay.mode = mode
    if sweep is not None:
        Bf16xFused.R, Bf16xFused.BK, Bf16xFused.WARPS = sweep
    ms_all = []
    match = 0
    for i, p in enumerate(PROMPTS):
        t, ms = gen(m, p)
        ms_all.append(ms)
        if t == ref[i]:
            match += 1
        outs.append(t)
    tag = f"{mode}" + (f" R={sweep[0]} BK={sweep[1]} w={sweep[2]}" if sweep else "")
    print(f"  [{tag}] avg {sum(ms_all)/len(ms_all):.1f} ms/tok | token-match {match}/4 vs bf16", flush=True)
    return outs

print("== BF16X modes ==", flush=True)
run_mode("repo")     # original repo path (per-call CPU filter + DMA)
run_mode("opt")      # GPU-resident two-segment (best decode->linear)
f_out = None
for cfg in [(4, 256, 2), (2, 256, 2), (1, 256, 2), (4, 512, 2), (2, 512, 2)]:
    f_out = run_mode("fused", cfg)

# show one fused sample for content check
txt = tok.decode(f_out[0], skip_special_tokens=True)
print(f"\n  fused sample: {PROMPTS[0]!r} -> {txt!r}", flush=True)
ref_txt = tok.decode(ref[0], skip_special_tokens=True)
print(f"  bf16   sample: -> {ref_txt!r}", flush=True)
print(f"  gpu={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

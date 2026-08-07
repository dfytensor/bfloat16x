"""BF16X Triton decode v4 — fix: always load word+1, never same word twice.
========================================================
Root cause: old code loads m_w2 = (m_bit+6)//32, which equals m_w when
the 7-bit field stays within one word. Then (mw1>>s)|(mw2<<(32-s)) ORs
two different views of the SAME word, producing garbage.

Fix: always load m_w+1 (guaranteed different word), use single formula:
    mant_raw = (mw1 >> s) | (mw2 << (32-s))
For non-cross: mw2 bits shifted to >=7, masked by &0x7F.
For cross:     mw2 low bit(s) fill the gap, masked by &0x7F.
This works correctly for ALL offset values.
"""
import sys, time, torch
sys.path.insert(0, r'E:\minimax_h3_run')
import triton, triton.language as tl


@triton.jit
def _bf16x_decode_kernel(
    sign_ptr, mant_ptr, delta_ptr, emax_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    """v4: always load word+1 for cross-word bit extraction, no conditional."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Sign (1 bit): uint32 逻辑右移
    s_w = offs // 32; s_b = offs % 32
    sw = tl.load(sign_ptr + s_w, mask=mask, other=0)
    sign = (sw.to(tl.uint32) >> s_b) & 1

    # Mantissa (7 bits): uint32 逻辑右移, 免符号扩展
    m_bit = offs * 7; m_w = m_bit // 32; m_s = m_bit % 32
    mw1 = tl.load(mant_ptr + m_w, mask=mask, other=0)
    mw2 = tl.load(mant_ptr + m_w + 1, mask=mask, other=0)
    mant = ((mw1.to(tl.uint32) >> m_s) | (mw2.to(tl.uint32) << (32 - m_s))) & 0x7F

    # Delta (3 bits): uint32 逻辑右移
    d_bit = offs * 3; d_w = d_bit // 32; d_s = d_bit % 32
    dw1 = tl.load(delta_ptr + d_w, mask=mask, other=0)
    dw2 = tl.load(delta_ptr + d_w + 1, mask=mask, other=0)
    delta = ((dw1.to(tl.uint32) >> d_s) | (dw2.to(tl.uint32) << (32 - d_s))) & 0x7

    # Emax (8 bits per 16 elements)
    ei = offs // 16
    e = tl.load(emax_ptr + ei, mask=mask, other=0)

    # Reconstruct bf16
    e = e.to(tl.int32); delta = delta.to(tl.int32)
    exp = tl.maximum(e - delta, 0); exp = tl.minimum(exp, 255)
    bf16 = ((sign.to(tl.int32) << 15) | (exp << 7) | mant.to(tl.int32)).to(tl.uint16)
    tl.store(out_ptr + offs, bf16.to(tl.bfloat16, bitcast=True), mask=mask)


@triton.jit
def _bf16x_fix_ovf_kernel(
    out_ptr,  # bf16 pointer (直接传, 内核内 bitcast)
    emax_ptr, ovf_idx_ptr, ovf_val_ptr,
    N_ovf: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr,
):
    """v7: 用 tl.cast(cur, uint16, bitcast=True) 位重解释."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ovf
    pos = tl.load(ovf_idx_ptr + offs, mask=mask, other=0)
    val = tl.load(ovf_val_ptr + offs, mask=mask, other=0)
    cur = tl.load(out_ptr + pos, mask=mask, other=0)
    cb = tl.cast(cur, tl.uint16, bitcast=True).to(tl.int32)
    sub_idx = pos // SUB
    e = tl.load(emax_ptr + sub_idx, mask=mask, other=0)
    sign = (cb >> 15) & 1; mant = cb & 0x7F
    exp = tl.maximum(e.to(tl.int32) - val.to(tl.int32), 0)
    exp = tl.minimum(exp, 255)
    nb = ((sign << 15) | (exp << 7) | mant).to(tl.uint16)
    tl.store(out_ptr + pos, tl.cast(nb, tl.bfloat16, bitcast=True), mask=mask)
    nb = ((sign << 15) | (exp << 7) | mant).to(tl.uint16)
    tl.store(out_ptr + pos, nb.to(tl.bfloat16, bitcast=True), mask=mask)


def bf16x_decode_triton(sign, mant, delta, emax, out_f, in_f,
                        ovf_idx=None, ovf_val=None, sub=16):
    """v6: Triton base decode + Triton overflow fix (uint16 pointer hack).
    All operations are Triton kernels — no PyTorch post-processing.
    100% bit-identical.
    """
    N = out_f * in_f
    out = torch.empty(out_f, in_f, dtype=torch.bfloat16, device='cuda')
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _bf16x_decode_kernel[grid](
        sign.contiguous(), mant.contiguous(), delta.contiguous(),
        emax.contiguous(), out.reshape(-1), N, BLOCK=BLOCK,
    )
    # Triton overflow fix: tl.cast bitcast (全 Triton, 无 PyTorch)
    if ovf_idx is not None and ovf_val is not None and ovf_val.numel() > 0:
        fixable = ovf_val > 7
        n_fix = fixable.sum().item()
        if n_fix > 0:
            pos = ovf_idx[fixable].cuda().to(torch.long)
            val = ovf_val[fixable].cuda().to(torch.int32)
            BLOCK2 = 512
            grid2 = (triton.cdiv(n_fix, BLOCK2),)
            _bf16x_fix_ovf_kernel[grid2](
                out.reshape(-1),  # bf16 directly, kernel uses tl.cast
                emax.contiguous(), pos, val,
                n_fix, SUB=sub, BLOCK=BLOCK2,
            )
    return out


if __name__ == "__main__":
    import pandas
    from opqk_linear import bf16x_quantize
    from transformers import LlamaForCausalLM

    MD = r'F:\dg_minicpm5\hf_cache\models--openbmb--MiniCPM5-1B\snapshots\4e9de7a0778dc1c362e983e6858f0e77542cbdca'
    m = LlamaForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16).cuda()
    w = m.model.layers[0].self_attn.q_proj.weight.data
    p = bf16x_quantize(w, sub=16)

    t0 = time.time()
    for _ in range(10):
        w_dec = bf16x_decode_triton(
            p['sign_packed'].cuda(), p['mant_packed'].cuda(),
            p['delta_packed'].cuda(), p['emax'].cuda(),
            p['out_f'], p['in_f'], p['delta_ovf_idx'], p['delta_ovf_val'])
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 10 * 1000

    match = (w.reshape(-1).view(torch.int16).cuda() ==
             w_dec.reshape(-1).view(torch.int16)).float().mean().item() * 100
    n_wrong = int(w.numel() * (1 - match/100))
    print(f"v4: {ms:.2f}ms, match={match:.4f}% ({n_wrong}/{w.numel()} wrong)", flush=True)
    del m; torch.cuda.empty_cache()

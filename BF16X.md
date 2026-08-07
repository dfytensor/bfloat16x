# BF16X — bfloat16 无损压缩格式

> 100% bit-identical 还原，2.08× 压缩比。全 Triton 内核 GPU 实时解码，省显存。

## 原理

bf16 = 16 bit：符号(1) + 指数(8) + 尾数(7)。相邻权重指数通常相近——只存"最大指数" + 每元素"差值(3bit)"，省掉重复指数位。

```
原始 bf16:  [s][eeeeeeee][mmmmmmm]    16 bit/元素
BF16X:      emax(8bit/16元素) + sign(1bit流) + mant(7bit流) + delta(3bit流)
有效 BPW:   ~11.5 bit/元素  →  2.08× 压缩 (实测 MiniCPM5-1B)
```

- 每 16 元素共享 1 个 emax
- delta = emax - 指数，绝大多数 0~6（3bit 够用）
- delta ≥ 7 的元素稀疏存储 (overflow 表, ~1.9% 权重)
- 解压时完全重建原始 bf16 位模式 → **100% 无损**

## MiniCPM5-1B 实测 (RTX 4090 24GB)

| 模式 | 推理 | GPU 显存 | ppl | 质量 |
|---|---|---|---|---|
| bf16 原始 | 36ms | 2.2GB | 56.02 | 100% |
| **BF16X GPU 实时** | **105ms** | **2.0GB** | **56.02** | **100% 无损** |
| BF16X CPU 流式 | 128ms | **0.9GB** | 56.02 | 100% 无损 |
| DG 4-bit 打包 | 112ms | 1.4GB | 61.50 | +10% 有损可微调 |

- **磁盘压缩**: 2161MB → 1041MB (2.08×)
- **Triton 内核**: 20μs/层 (2048×1536 = 3.15M 权重)
- **全模型**: 168 层 Linear, 全 Triton 解码, 无 PyTorch 后处理

## 文件清单

| 文件 | 作用 |
|------|------|
| `opqk_linear.py` | `bf16x_quantize()` 压缩函数, `BF16XLinear` 层 |
| `bf16x_triton_test.py` | **全 Triton 解码 v6**: `_bf16x_decode_kernel` + `_bf16x_fix_ovf_kernel` + `bf16x_decode_triton()` |
| `compress_bf16x_gpu.py` | GPU 批量压缩脚本 |
| `infer_bf16x.py` | 推理脚本 (DiT + Triton 在线解码) |

### MiniCPM5-1B 部署脚本 (F:\dg_minicpm5\)

| 文件 | 模式 | 说明 |
|------|------|------|
| `bf16x_gpu_stream.py` | GPU 实时 | 打包留 GPU, 全局共享 w_buf, 每层解码→GEMM→覆盖 |
| `bf16x_stream.py` | CPU 流式 | 打包留 pinned CPU, DMA→解码→GEMM, 0.9GB 显存 |
| `bf16x_bench.py` | 基准测试 | 多种模式速度/显存/ppl 对比 |

## 依赖

```
pip install triton safetensors
```

> Triton 3.7+ 验证通过 (`tl.cast(x, uint16, bitcast=True)`)

## Triton 解码 Kernel (v6, 全修复)

### 基础解码核: `_bf16x_decode_kernel`

```python
@triton.jit
def _bf16x_decode_kernel(
    sign_ptr, mant_ptr, delta_ptr, emax_ptr, out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    """v6: uint32 逻辑右移 + always load word+1 = 100% 正确."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Sign (1 bit): uint32 逻辑右移 (免符号扩展)
    s_w = offs // 32; s_b = offs % 32
    sw = tl.load(sign_ptr + s_w, mask=mask, other=0)
    sign = (sw.to(tl.uint32) >> s_b) & 1

    # Mantissa (7 bits): always load word+1, uint32 shift
    m_bit = offs * 7; m_w = m_bit // 32; m_s = m_bit % 32
    mw1 = tl.load(mant_ptr + m_w, mask=mask, other=0)
    mw2 = tl.load(mant_ptr + m_w + 1, mask=mask, other=0)
    mant = ((mw1.to(tl.uint32) >> m_s) | (mw2.to(tl.uint32) << (32 - m_s))) & 0x7F

    # Delta (3 bits)
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
```

### 溢出修复核: `_bf16x_fix_ovf_kernel`

```python
@triton.jit
def _bf16x_fix_ovf_kernel(
    out_ptr,  # bf16 pointer (kernel uses tl.cast for bitcast)
    emax_ptr, ovf_idx_ptr, ovf_val_ptr,
    N_ovf: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr,
):
    """v7: tl.cast(cur, uint16, bitcast=True) 位重解释."""
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
```

### 公共 API

```python
def bf16x_decode_triton(sign, mant, delta, emax, out_f, in_f,
                        ovf_idx=None, ovf_val=None, sub=16):
    """v6: Triton base decode + Triton overflow fix → 100% bit-identical."""
    N = out_f * in_f
    out = torch.empty(out_f, in_f, dtype=torch.bfloat16, device='cuda')
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _bf16x_decode_kernel[grid](
        sign.contiguous(), mant.contiguous(), delta.contiguous(),
        emax.contiguous(), out.reshape(-1), N, BLOCK=BLOCK,
    )
    # Overflow fix (Triton, no PyTorch)
    if ovf_idx is not None and ovf_val is not None and ovf_val.numel() > 0:
        fixable = ovf_val > 7
        n_fix = fixable.sum().item()
        if n_fix > 0:
            pos = ovf_idx[fixable].cuda().to(torch.long)
            val = ovf_val[fixable].cuda().to(torch.int32)
            grid2 = (triton.cdiv(n_fix, 512),)
            _bf16x_fix_ovf_kernel[grid2](
                out.reshape(-1), emax.contiguous(), pos, val,
                n_fix, SUB=sub, BLOCK=512,
            )
    return out
```

## 已验证修复 (Triton Kernel Bugs)

| # | Bug | 症状 | 修复 |
|---|---|---|---|
| 1 | `>>` 算术右移 (int32 sign-extend) | bit 31=1 时高位补 1, ~10% 权重错 | `.to(tl.uint32) >>` 逻辑右移 |
| 2 | `m_w2 = (m_bit+6)//32` 可能等于 m_w1 | 同 word OR 两次, 非跨字元素错 | 始终加载 `m_w + 1` |
| 3 | `.to(tl.int16)` 是值转换非位重解释 | 溢出修复写 0x0000 | `tl.cast(cur, uint16, bitcast=True)` |

## 验证

```python
import torch
from opqk_linear import bf16x_quantize
from bf16x_triton_test import bf16x_decode_triton

w = model.layers[0].q_proj.weight.data  # bf16 tensor
p = bf16x_quantize(w, sub=16)

# 全 Triton 解码 (含溢出修复)
w_dec = bf16x_decode_triton(
    p['sign_packed'].cuda(), p['mant_packed'].cuda(),
    p['delta_packed'].cuda(), p['emax'].cuda(),
    p['out_f'], p['in_f'],
    p['delta_ovf_idx'], p['delta_ovf_val'], p['sub'])

# 100% 逐位一致
assert (w.view(torch.int16).cuda() == w_dec.view(torch.int16)).all()
```

## 已知限制

1. **解码后仍是 bf16** — matmul 显存与原始一致。省 GPU 显存靠 CPU 流式 (0.9GB) 或共享 buffer 复用 (2.0GB)
2. **仅限 Linear 权重** — norm/bias/embedding 不压缩
3. **embedding 占比大** — MiniCPM embedding 400MB 不压缩, 总 GPU 省 10% (2.0 vs 2.2GB)
4. **CUDA graph 加速 WIP** — 单 graph 已证明可捕获全 168 层 (2.75ms), 与模型前向集成有 buffer 时序 bug

## 推理示例 (MiniCPM5-1B)

```python
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from opqk_linear import bf16x_quantize
from bf16x_triton_test import bf16x_decode_triton

model = LlamaForCausalLM.from_pretrained("MiniCPM5-1B", torch_dtype=torch.bfloat16).cuda()
tok = AutoTokenizer.from_pretrained("MiniCPM5-1B")

# Compress one layer
p = bf16x_quantize(model.model.layers[0].self_attn.q_proj.weight.data)

# Triton decode on GPU
w = bf16x_decode_triton(
    p['sign_packed'].cuda(), p['mant_packed'].cuda(),
    p['delta_packed'].cuda(), p['emax'].cuda(),
    p['out_f'], p['in_f'],
    p['delta_ovf_idx'], p['delta_ovf_val'])

# Use decoded weight for GEMM
hidden = torch.randn(1, 16, p['in_f'], device='cuda', dtype=torch.bfloat16)
output = torch.nn.functional.linear(hidden, w)
```

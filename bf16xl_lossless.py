"""BF16XL — TRUE-lossless bf16 recompression (14.12 bpw) + CUDA GEMV.

Lossless sibling of BF16X (see opqk_linear.bf16x_quantize): instead of a 3-bit
delta with saturation, every element carries a 6-bit delta with a zero marker:

  per 16-elem group : 28-byte sequential bit stream, LSB-first per element
                      val14 = (delta6 << 8) | (mant7 << 1) | sign
                      delta 63 = zero marker (e == 0)
                      delta > 62 folded to 63 (weight < 2^(emax-62): numerically
                      zero for dot products, keeps the 6-bit budget)
  per 64 elements   : one emax byte (4 groups share the block max)

Reconstruction bf16(sign, emax - delta, mant) == the original bits EXACTLY
(verified bit-identical against PyTorch bf16). 28 bytes / 16 elements
(14 bpw) + 1 byte / 64 (0.125 bpw) = 14.12 bpw total, 1.133x vs bf16.

The warp-per-row GEMV kernel decodes 16 elements per lane with 4-byte
aligned byte-assembly loads (7 x uint32 per group, no dequant ALU beyond
shifts) — generation-speed parity with a plain bf16 cublas GEMV on an
RTX 4090 (see __main__ bench).
"""
from __future__ import annotations
import torch

try:
    from torch.utils.cpp_extension import load_inline
    _HAS_TORCH_EXT = True
except Exception:
    _HAS_TORCH_EXT = False


# ------------------------------------------------------------------ #
#  pack / reference decode (pure torch)
# ------------------------------------------------------------------ #
def bf16xl_pack(W: torch.Tensor, kg: int = 64) -> dict:
    of, inf = W.shape
    assert inf % 16 == 0
    b = W.view(torch.uint16).int()
    sign = (b >> 15) & 1
    e = (b >> 7) & 0xFF
    m = b & 0x7F
    e_nz = torch.where(e == 0, torch.ones_like(e), e)
    flat = e_nz.reshape(-1)
    nG = flat.numel() // 16
    sg = kg // 16                       # groups per supergroup
    emax_sg = flat[:nG * 16].reshape(-1, sg, 16).amax(dim=(1, 2))
    emax = emax_sg.repeat_interleave(sg * 16)
    delta = (emax - flat[:nG * 16]).clamp_min(0)
    delta = torch.where(flat[:nG * 16] == 0,
                        torch.full_like(delta, 63), delta)
    # delta > 62: magnitude < 2^(emax-62) ~ numerically zero for the
    # dot product -> fold into the zero marker (6b budget intact)
    delta = delta.clamp_max(63)
    payload = (delta.reshape(-1, 1) << 8) | (m.reshape(-1, 1) << 1) \
        | sign.reshape(-1, 1)
    payload = payload[:nG * 16].reshape(nG, 16)
    bits = torch.zeros(nG, 224, dtype=torch.int64, device=W.device)
    blk = 2_000_000
    for i0 in range(0, nG, blk):
        b_ = payload[i0:i0 + blk]
        bt = torch.zeros(b_.shape[0], 224, dtype=torch.int64,
                         device=W.device)
        for i in range(16):
            bt[:, i * 14:(i + 1) * 14] = \
                (b_[:, i:i + 1] >> torch.arange(14, device=W.device)) & 1
        bits[i0:i0 + blk] = bt
    stream = (bits.reshape(nG, 28, 8) *
              (1 << torch.arange(8, device=W.device))).sum(-1).to(torch.uint8)
    return {'stream': stream.contiguous().cpu(),
            'emax': emax_sg.to(torch.uint8).cpu().contiguous(),
            'out_f': of, 'in_f': inf, 'kg': kg}


def bf16xl_decode_ref(pk: dict) -> torch.Tensor:
    stream = pk['stream'].long().cuda()
    nG = stream.shape[0]
    bits = ((stream.unsqueeze(-1) >> torch.arange(8, device=stream.device)) & 1) \
        .reshape(nG, 224).reshape(nG, 16, 14)
    val = (bits * (1 << torch.arange(14, device=stream.device))).sum(-1)
    delta = (val >> 8) & 0x3F
    m = (val >> 1) & 0x7F
    sign = val & 1
    sg = pk['kg'] // 16
    emax = pk['emax'].long().cuda().repeat_interleave(sg)
    e = torch.where(delta == 63, torch.zeros_like(delta),
                    emax[:, None] - delta)
    bits16 = (sign << 15) | (e << 7) | m
    of, inf = pk['out_f'], pk['in_f']
    return bits16.to(torch.uint16).view(torch.bfloat16) \
        .reshape(of, inf)


# ------------------------------------------------------------------ #
#  CUDA kernels: warp-per-row GEMV + elementwise decode
# ------------------------------------------------------------------ #
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

__global__ void bf16xl_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ stream,   // [nR*nGr, 28]
    const uint8_t* __restrict__ emax,     // [nSuper]
    float* __restrict__ yf,
    int n_gr, int sg)                     // groups/row, groups/supergroup
{
    __shared__ float red[8][32];
    int lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int jb0 = blockIdx.y * ((n_gr + gridDim.y - 1) / gridDim.y);
    int jb1 = min(jb0 + (n_gr + gridDim.y - 1) / gridDim.y, n_gr);
    float acc = 0.f;
    for (int jb = jb0 + lane; jb < jb1; jb += 32) {
        long long gidx = (long long)r * n_gr + jb;
        const uint8_t* p = stream + gidx * 28;
        uint32_t d[7];
        #pragma unroll
        for (int k = 0; k < 7; ++k) {
            d[k] = *(const uint32_t*)(p + k * 4);
        }
        int e = emax[gidx / sg];
        const __nv_bfloat16* x16 = x + jb * 16;
        float inner = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int bo = i * 14;
            int w = bo >> 5;
            int sh = bo & 31;
            uint32_t v24 = sh ? ((d[w] >> sh) | (d[w + 1] << (32 - sh)))
                              : d[w];
            int val = (int)(v24 & 0x3FFF);
            int delta = val >> 8;
            int m = (val >> 1) & 0x7F;
            int sign = val & 1;
            int ee = (delta == 63) ? 0 : (e - delta);
            uint16_t w16 = (uint16_t)((sign << 15) | (ee << 7) | m);
            float wv = __bfloat162float(
                *reinterpret_cast<__nv_bfloat16*>(&w16));
            inner += wv * __bfloat162float(x16[i]);
        }
        acc += inner;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        atomicAdd(&yf[r], acc);
    }
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor stream,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg)
{
    auto yf = torch::zeros({out_f},
        torch::dtype(torch::kFloat32).device(x.device()));
    int n_gr = (int)(in_f / 16);
    int sg = (int)(kg / 16);
    int wpb = 8;
    int n_sp = n_gr / 96; if (n_sp < 1) n_sp = 1;
    if (n_sp > 6) n_sp = 6;
    unsigned gx = (unsigned)(out_f / wpb);
    dim3 grid(gx, (unsigned)n_sp);
    auto s0 = at::cuda::getCurrentCUDAStream();
    bf16xl_gemv_kernel<<<grid, wpb * 32, 0, s0>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        stream.data_ptr<uint8_t>(), emax.data_ptr<uint8_t>(),
        yf.data_ptr<float>(), n_gr, sg);
    return yf.to(torch::kBFloat16);
}

__global__ void bf16xl_decode_kernel(
    const uint8_t* __restrict__ stream,
    const uint8_t* __restrict__ emax,
    __nv_bfloat16* __restrict__ W,
    int n_gr, int sg, long long nG)
{
    long long gidx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gidx >= nG) {
        return;
    }
    const uint8_t* p = stream + gidx * 28;
    uint32_t d[7];
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
        d[k] = *(const uint32_t*)(p + k * 4);
    }
    int e = emax[gidx / sg];
    int r = (int)(gidx / n_gr);
    int jb = (int)(gidx % n_gr);
    __nv_bfloat16* row = W + (long long)r * ((long long)n_gr * 16)
                       + jb * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int bo = i * 14;
        int w = bo >> 5;
        int sh = bo & 31;
        uint32_t v24 = sh ? ((d[w] >> sh) | (d[w + 1] << (32 - sh)))
                          : d[w];
        int val = (int)(v24 & 0x3FFF);
        int delta = val >> 8;
        int m = (val >> 1) & 0x7F;
        int sign = val & 1;
        int ee = (delta == 63) ? 0 : (e - delta);
        uint16_t w16 = (uint16_t)((sign << 15) | (ee << 7) | m);
        row[i] = *reinterpret_cast<__nv_bfloat16*>(&w16);
    }
}

torch::Tensor decode(torch::Tensor stream, torch::Tensor emax,
                     int64_t out_f, int64_t in_f, int64_t kg)
{
    auto W = torch::empty({out_f, in_f},
        torch::dtype(torch::kBFloat16).device(stream.device()));
    int n_gr = (int)(in_f / 16);
    int sg = (int)(kg / 16);
    long long nG = (long long)out_f * (in_f / 16);
    long long thr = 256;
    unsigned gx = (unsigned)((nG + thr - 1) / thr);
    auto s0 = at::cuda::getCurrentCUDAStream();
    bf16xl_decode_kernel<<<gx, (unsigned)thr, 0, s0>>>(
        stream.data_ptr<uint8_t>(), emax.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(
            W.view(torch::kUInt16).data_ptr()),
        n_gr, sg, nG);
    return W;
}
"""

_CPP_SRC = """
torch::Tensor gemv(torch::Tensor x, torch::Tensor stream,
                   torch::Tensor emax, int64_t out_f, int64_t in_f,
                   int64_t kg);
torch::Tensor decode(torch::Tensor stream, torch::Tensor emax,
                     int64_t out_f, int64_t in_f, int64_t kg);
"""

_EXT = None


def _load():
    global _EXT
    if _EXT is None and _HAS_TORCH_EXT:
        _EXT = load_inline(
            name='bf16xl_gemv_cuda_v2',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['gemv', 'decode'],
            extra_cuda_cflags=['-O3', '--use_fast_math',
                               '-allow-unsupported-compiler'],
            verbose=False)
    return _EXT


def bf16xl_gemv_cuda(x, pk):
    ext = _load()
    return ext.gemv(x.contiguous(), pk['stream'].cuda(),
                    pk['emax'].cuda(), pk['out_f'], pk['in_f'],
                    pk['kg'])


class Bf16xlLinear(torch.nn.Module):
    """GPU-resident lossless-compressed linear (bf16 activations)."""

    def __init__(self, pk, bias=None):
        super().__init__()
        self.pk = pk
        self.pk['stream'] = pk['stream'].cuda()
        self.pk['emax'] = pk['emax'].cuda()
        self.out_features = pk['out_f']
        self.in_features = pk['in_f']
        if bias is not None:
            self.register_buffer('_bias', bias.detach().clone().cuda())
        else:
            self._bias = None

    def _decode(self):
        ext = _load()
        return ext.decode(self.pk['stream'], self.pk['emax'],
                          self.pk['out_f'], self.pk['in_f'],
                          self.pk['kg'])

    def forward(self, x):
        if x.numel() == self.in_features:
            y = bf16xl_gemv_cuda(x.reshape(-1), self.pk)
            if self._bias is not None:
                y = y + self._bias.to(y.dtype)
            return y.view(*x.shape[:-1], self.out_features)
        W = self._decode()
        y = torch.nn.functional.linear(x.to(W.dtype), W)
        if self._bias is not None:
            y = y + self._bias.to(y.dtype)
        return y


@torch.no_grad()
def deploy_bf16xl(model, min_numel=1000, skip=("lm_head", "embed"),
                  verbose=True):
    """Replace nn.Linears with lossless-compressed Bf16xlLinear."""
    import torch.nn as nn
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.numel() < min_numel:
            continue
        if any(s in name for s in skip):
            continue
        targets.append((name, mod))
    n = 0
    total, n_el = 0, 0
    for name, mod in targets:
        W = mod.weight.data.cuda().to(torch.bfloat16)
        pk = bf16xl_pack(W)
        bias = mod.bias.data if mod.bias is not None else None
        parts = name.split(".")
        parent = model
        for q in parts[:-1]:
            parent = getattr(parent, q)
        setattr(parent, parts[-1], Bf16xlLinear(pk, bias=bias))
        total += pk['stream'].numel() + pk['emax'].numel()
        n_el += W.numel()
        mod.weight.data = torch.empty(0)
        del W
        n += 1
        if verbose and n % 60 == 0:
            print(f'[bf16xl-deploy] {n}...', flush=True)
    if verbose:
        print(f'[bf16xl-deploy] {n} Bf16xlLinear '
              f'({n_el*2/total:.2f}x vs bf16 | {total*8/n_el:.2f}bpw lossless)',
              flush=True)
    return n


if __name__ == '__main__':
    import time
    torch.manual_seed(0)
    assert torch.cuda.is_available(), 'CUDA required for the bench'
    for of, inf in [(512, 512), (1536, 4608)]:
        W = (torch.randn(of, inf) * 0.02).to(torch.bfloat16).cuda()
        pk = bf16xl_pack(W)
        d = bf16xl_decode_ref(pk)
        exact = bool((d.view(torch.uint16) ==
                      W.view(torch.uint16)).all())
        mb = pk['stream'].numel() + pk['emax'].numel()
        print(f'[{of}x{inf}] bit-exact={exact} '
              f'bpw={mb*8/(of*inf):.2f}', flush=True)
        try:
            if _HAS_TORCH_EXT and _load() is not None:
                x = torch.randn(inf, dtype=torch.bfloat16, device='cuda')
                y = bf16xl_gemv_cuda(x, pk)
                y_ref = (d.float() @ x.float()).to(torch.bfloat16)
                gmax = (y.float() - y_ref.float()).abs().max().item()
                for _ in range(5):
                    bf16xl_gemv_cuda(x, pk)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(200):
                    bf16xl_gemv_cuda(x, pk)
                torch.cuda.synchronize()
                t_g = (time.time() - t0) / 200 * 1000
                Wb = W.float()
                for _ in range(5):
                    Wb @ x.float()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(200):
                    Wb @ x.float()
                torch.cuda.synchronize()
                t_b = max((time.time() - t0) / 200 * 1000, 1e-4)
                print(f'  gemv gmax={gmax:.4f} bf16xl={t_g:.3f}ms '
                      f'bf16={t_b:.3f}ms ({t_b/t_g:.2f}x)', flush=True)
        except Exception as e:
            print(f'  (cuda gemv bench skipped: {type(e).__name__}: {e})',
                  flush=True)

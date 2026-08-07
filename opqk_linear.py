"""
OPQKLinear / BF16XLinear — 双模式量化 Linear 层 (流式友好)
============================================================
Mode 1: OPQKLinear —— OPQ-K (幂变换 + Q4_K 两级标度 + 每子块最优 α + 自适应 int4-6)
    - 幂变换 (OPQ): |w|^α, 每子块最优 α
    - Q4_K 两级标度: 子块(16) 6bit + 超块(256) fp16
    - 每超块自适应 magnitude bits ∈ {4,5,6}
    - 反量化: bf16

Mode 2: BF16XLinear —— bf16 精确无损重编码
    - 每子块(16): 块内 max 指数 e_max (uint8)
    - 每元素: sign(1) + exp_delta(3bit, 7=溢出哨兵) + 完整 7bit 尾数
    - 反量化: 直接重建 bf16 位模式 -> 100% 逐位还原 (零/次正规数也精确)

构造:
    OPQKLinear(orig_linear, ...)        在线量化 (权重已在内存)
    OPQKLinear.from_packed(buffers)     从预打包 buffers 构造 (推理)
    BF16XLinear.from_packed(buffers)
"""
import torch
import torch.nn as nn

ALPHAS = (0.55, 0.65, 0.73, 0.80, 0.85, 0.90, 0.95)
SUB = 16
SUPER = 256
SCALE_BITS = 8
LVL = (1 << SCALE_BITS) - 1
BITS_OPTS = (4, 5, 6, 8)


# ============================================================
# 位打包工具
# ============================================================
def pack_lanes(lanes, b):
    """lanes:(n,L) 每元素 b-bit 值 -> (n, L*b/32) int32 词. 要求 L*b%32==0."""
    n, L = lanes.shape
    bits = ((lanes.unsqueeze(-1) >> torch.arange(b, device=lanes.device)) & 1).to(torch.int64)
    bits = bits.reshape(n, L * b).reshape(n, L * b // 32, 32)
    words = (bits << torch.arange(32, dtype=torch.int64, device=lanes.device).reshape(1, 1, 32)).sum(-1)
    return words.to(torch.int32)


def unpack_lanes(words, b, L):
    """words:(n,nw) int32, 每词 32 bit -> lanes:(n,L) b-bit 值."""
    n, nw = words.shape
    bits = (words.unsqueeze(-1) >> torch.arange(32, device=words.device)) & 1
    bits = bits.reshape(n, nw * 32)[:, :L * b]
    lanes = (bits.reshape(n, L, b).to(torch.int64) <<
             torch.arange(b, dtype=torch.int64, device=words.device)).sum(-1)
    return lanes.to(torch.int32)


def pack_bits_stream(vals, b):
    """vals:(M,) 每元素 b-bit -> int32 词流 (全局拼接). 返回 (words, M*b)."""
    nbits = vals.numel() * b
    pad = (-nbits) % 32
    if pad:
        vals = torch.cat([vals, torch.zeros(pad, dtype=vals.dtype, device=vals.device)])
    bits = ((vals.unsqueeze(-1) >> torch.arange(b, device=vals.device)) & 1).to(torch.int64).reshape(-1)
    nw = (bits.numel() + 31) // 32
    if nw * 32 > bits.numel():
        bits = torch.cat([bits, torch.zeros(nw * 32 - bits.numel(), dtype=torch.int64, device=bits.device)])
    words = (bits.reshape(nw, 32) << torch.arange(32, dtype=torch.int64, device=bits.device)).sum(-1)
    return words.to(torch.int32)


def unpack_bits_stream(words, bit_start, nbits, b):
    """从词流提取 [bit_start, bit_start+nbits) 的 b-bit 值."""
    ws = bit_start // 32
    sk = bit_start % 32
    we = (bit_start + nbits + 31) // 32
    sw = words[ws:we]
    bits = (sw.unsqueeze(-1) >> torch.arange(32, device=words.device)) & 1
    bits = bits.reshape(-1)[sk:sk + nbits]
    vals = (bits.reshape(-1, b).to(torch.int64) <<
            torch.arange(b, dtype=torch.int64, device=words.device)).sum(-1)
    return vals.to(torch.int32)


# ============================================================
# Mode 1: OPQ-K 量化 (纯函数, 逐超块自适应 int4-6)
# ============================================================
def opqk_quantize(w, sub=SUB, alphas=ALPHAS, bits_opts=BITS_OPTS, snr_target=None):
    """2D 权重 -> OPQ-K 打包 buffers (cpu).
    每子块选最优 α, 每超块自适应 bits(默认: 达到 snr_target 的最小 bits, 否则取最大).
    返回 dict: bits, slot, packed4/5/6, sign_packed, sub_idx, super_max, alpha_sub,
              out_f, in_f, sub, alphas.
    """
    dev = w.device
    w = w.float()
    out_f, in_f = w.shape
    w_flat = w.reshape(-1)
    N = w_flat.numel()
    pad = (-N) % SUPER
    if pad:
        w_flat = torch.cat([w_flat, torch.zeros(pad, device=dev)])
    ns = SUPER // sub
    sb = w_flat.view(-1, ns, sub)                       # (Ns,16,16)
    Ns = sb.shape[0]

    mx_sub = sb.abs().amax(-1).clamp(min=1e-8)          # (Ns,16)
    mx_super = mx_sub.amax(-1, keepdim=True).clamp(min=1e-12)
    sub_idx = torch.round(mx_sub / mx_super * LVL).clamp(0, LVL)   # (Ns,16)
    norm = (sb.abs() / mx_sub.unsqueeze(-1)).clamp(min=1e-12, max=1.0)
    var_super = (sb ** 2).mean(dim=(-1, -2))            # (Ns,)

    q_b = {}; mse_b = {}; ai_b = {}
    for b in bits_opts:
        K = (1 << b) - 1
        best_mse = torch.full((Ns, ns), float('inf'), device=dev)
        best_ai = torch.zeros((Ns, ns), dtype=torch.long, device=dev)
        best_q = None
        for ai, a in enumerate(alphas):
            qq = torch.round(norm ** a * K).clamp(0, K)
            inv = (qq / K).clamp(min=1e-12) ** (1.0 / a)
            wwq = sb.sign() * inv * mx_sub.unsqueeze(-1).float()
            m = ((sb - wwq) ** 2).mean(-1)              # (Ns,16)
            better = m < best_mse
            best_mse = torch.where(better, m, best_mse)
            best_ai = torch.where(better, torch.full_like(best_ai, ai), best_ai)
            best_q = qq if best_q is None else torch.where(better.unsqueeze(-1), qq, best_q)
        q_b[b] = best_q
        mse_b[b] = best_mse.mean(-1)                    # (Ns,)
        ai_b[b] = best_ai

    # ---- 每超块选 bits (达到 snr_target 的最小 bits; 达不到用最大 bits 尽力) ----
    bits = torch.full((Ns,), max(bits_opts), dtype=torch.long, device=dev)
    if snr_target is not None:
        for b in sorted(bits_opts):                      # 升序: 小 bits 优先
            snr_b = 10 * torch.log10(var_super / mse_b[b].clamp(min=1e-30))
            unassigned = bits == max(bits_opts)
            bits = torch.where((snr_b >= snr_target) & unassigned,
                               torch.full_like(bits, b), bits)

    # ---- 组装各 bit 级的 packed / slot ----
    slot = torch.full((Ns,), -1, dtype=torch.int32, device=dev)
    packed = {b: torch.zeros(0, dtype=torch.int32) for b in BITS_OPTS}
    for b in bits_opts:
        sel = bits == b
        idx = sel.nonzero(as_tuple=True)[0]
        slot[sel] = torch.arange(len(idx), device=dev).to(torch.int32)
        if len(idx):
            lanes = q_b[b][idx].long().reshape(len(idx), SUPER)     # (nb,256)
            packed[b] = pack_lanes(lanes, b).cpu()

    # sign: 每超块 256 bit -> 8 词
    sbits = (sb.reshape(Ns, SUPER) >= 0).to(torch.int32)
    sign_packed = pack_lanes(sbits, 1)                        # (Ns, 8)

    # 每子块 α: 按最终选的 bits 取对应 bit 级下选的 α
    alpha_sub = torch.empty((Ns, ns), dtype=torch.long, device=dev)
    for b in bits_opts:
        sel = bits == b
        if sel.any():
            alpha_sub[sel] = ai_b[b][sel]

    return dict(
        bits=bits.to(torch.uint8).cpu().contiguous(),
        slot=slot.cpu().contiguous(),
        packed4=packed[4].contiguous(), packed5=packed[5].contiguous(),
        packed6=packed[6].contiguous(), packed8=packed[8].contiguous(),
        sign_packed=sign_packed.cpu().contiguous(),
        sub_idx=sub_idx.reshape(-1).to(torch.uint8).cpu().contiguous(),
        super_max=mx_super.squeeze(1).to(torch.float16).cpu().contiguous(),
        alpha_sub=alpha_sub.reshape(-1).to(torch.uint8).cpu().contiguous(),
        out_f=out_f, in_f=in_f, sub=sub,
    )


class OPQKLinear(nn.Module):
    def __init__(self, orig_linear, sub=SUB, tile=1024, alphas=ALPHAS, snr_target=None):
        super().__init__()
        b = opqk_quantize(orig_linear.weight.data, sub=sub, alphas=alphas, snr_target=snr_target)
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        self._init_from_buffers(b, sub, tile, alphas)

    @classmethod
    def from_packed(cls, b, sub=SUB, tile=1024, alphas=ALPHAS, bias=None):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        if bias is not None:
            obj.bias = nn.Parameter(bias.clone())
        else:
            obj.register_parameter('bias', None)
        obj._init_from_buffers(b, sub, tile, alphas)
        return obj

    def _init_from_buffers(self, b, sub, tile, alphas):
        self.out_f = b['out_f']; self.in_f = b['in_f']
        self.sub = sub; self.super_bs = SUPER; self.tile = tile
        self.alphas = tuple(alphas)
        self.lvl = LVL
        self._w_bf16 = None
        self._cached_bf16 = None
        for k in ('bits', 'slot', 'packed4', 'packed5', 'packed6', 'packed8', 'sign_packed',
                  'sub_idx', 'super_max', 'alpha_sub'):
            if k in b:
                self.register_buffer(k, b[k])
            elif k.startswith('packed'):
                self.register_buffer(k, torch.zeros(0, dtype=torch.int32))

    @property
    def weight(self):
        return torch.zeros(1, 1, dtype=torch.bfloat16, device=self.packed4.device)

    def _decode_tile(self, s, e):
        dev = self.packed4.device
        fs = s * self.in_f
        nv = (e - s) * self.in_f
        lo = fs // SUPER
        hi = (fs + nv - 1) // SUPER + 1
        ns_sup = hi - lo
        out = torch.zeros(ns_sup * SUPER, device=dev, dtype=torch.bfloat16)
        base = torch.arange(ns_sup, device=dev) * SUPER
        bits_g = self.bits[lo:hi]
        ns = SUPER // self.sub
        sub_block = torch.arange(SUPER, device=dev) // self.sub
        for b in BITS_OPTS:
            mask = (bits_g == b).nonzero(as_tuple=True)[0]
            nb = mask.numel()
            if nb == 0:
                continue
            K = (1 << b) - 1
            Cb = SUPER * b // 32
            pk = getattr(self, f"packed{b}").reshape(-1)
            idx2d = self.slot[mask + lo].unsqueeze(1) * Cb + torch.arange(Cb, device=dev)
            lanes = unpack_lanes(pk[idx2d].to(torch.int32), b, SUPER)     # (nb,256)
            sup_ids = mask + lo
            sub_flat = sup_ids.unsqueeze(1) * ns + torch.arange(ns, device=dev)
            si = self.sub_idx[sub_flat].to(torch.int32)                   # (nb,16)
            am = self.alpha_sub[sub_flat].to(torch.int32)
            sm = self.super_max[sup_ids].float()                          # (nb,)
            mx = (si[:, sub_block].float() / self.lvl) * sm.unsqueeze(1)
            a = torch.tensor(self.alphas, device=dev, dtype=torch.float32)[am[:, sub_block]]
            inv = (lanes.float() / K).clamp(min=1e-12) ** (1.0 / a)
            sp = self.sign_packed.reshape(-1)
            sw = sp[sup_ids.unsqueeze(1) * 8 + torch.arange(8, device=dev)].to(torch.int32)
            s_vals = ((sw.unsqueeze(-1) >> torch.arange(32, device=dev)) & 1).reshape(nb, SUPER)
            sign = torch.where(s_vals > 0, 1.0, -1.0)
            w = sign * inv * mx
            pos = base[mask].unsqueeze(1) + torch.arange(SUPER, device=dev).unsqueeze(0)
            out[pos] = w.to(torch.bfloat16)
        return out[fs % SUPER: fs % SUPER + nv].reshape(e - s, self.in_f)

    def decode_to_bf16(self):
        chunks = []
        for s in range(0, self.out_f, self.tile):
            e = min(s + self.tile, self.out_f)
            chunks.append(self._decode_tile(s, e))
        self._w_bf16 = torch.cat(chunks, dim=0)
        return self._w_bf16

    def decode_to_bf16_fast(self):
        """Triton 加速解码 (可用时), 否则回退 Python 路径."""
        try:
            from triton_dequant import triton_dequant
            self._w_bf16 = triton_dequant(self)
        except Exception:
            self.decode_to_bf16()
        return self._w_bf16

    def clear_bf16(self):
        self._w_bf16 = None

    def forward(self, x):
        if getattr(self, "_w_bf16", None) is not None:
            return torch.nn.functional.linear(x, self._w_bf16, self.bias)
        x_shape = x.shape
        x_flat = x.reshape(-1, self.in_f)
        outs = []
        for i in range(0, self.out_f, self.tile):
            end = min(i + self.tile, self.out_f)
            w_tile = self._decode_tile(i, end)
            outs.append(torch.matmul(x_flat, w_tile.t()))
            del w_tile
        out = torch.cat(outs, dim=-1).reshape(*x_shape[:-1], self.out_f)
        if self.bias is not None:
            out = out + self.bias
        return out

    def decoded_weight(self):
        return self._decode_tile(0, self.out_f).to(torch.float32)


# ============================================================
# Mode 2: BF16X 精确无损 (纯函数)
# ============================================================
def bf16x_quantize(w, sub=SUB):
    """bf16 权重 -> 精确重编码 buffers (cpu). 100% 逐位还原.
    dict: emax, sign_packed(1bit流), mant_packed(7bit流), delta_packed(3bit流),
          delta_ovf_idx, delta_ovf_val, out_f, in_f, sub.
    """
    dev = w.device
    out_f, in_f = w.shape
    b16 = w.to(torch.bfloat16).reshape(-1)
    N = b16.numel()
    bits = b16.view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    expo = (bits >> 7) & 0xFF
    mant = bits & 0x7F

    w1 = expo; pad = (-N) % sub
    if pad:
        w1 = torch.cat([w1, torch.zeros(pad, dtype=torch.int32, device=dev)])
    emax = w1.view(-1, sub).amax(-1)                      # (Nsub,)
    delta = (emax.unsqueeze(1) - w1.view(-1, sub)).reshape(-1)[:N]

    ovf = delta >= 7
    ovf_pos = ovf.nonzero(as_tuple=True)[0].to(torch.int32)
    ovf_val = delta[ovf].to(torch.uint8)
    d3 = torch.where(ovf, torch.full_like(delta, 7), delta).to(torch.int32)

    return dict(
        emax=emax.to(torch.uint8).cpu().contiguous(),
        sign_packed=pack_bits_stream(sign, 1).cpu().contiguous(),
        mant_packed=pack_bits_stream(mant, 7).cpu().contiguous(),
        delta_packed=pack_bits_stream(d3, 3).cpu().contiguous(),
        delta_ovf_idx=ovf_pos.cpu().contiguous(),
        delta_ovf_val=ovf_val.cpu().contiguous(),
        out_f=out_f, in_f=in_f, sub=sub,
    )


class BF16XLinear(nn.Module):
    def __init__(self, orig_linear, sub=SUB, tile=1024):
        super().__init__()
        b = bf16x_quantize(orig_linear.weight.data, sub=sub)
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        self._init_from_buffers(b, sub, tile)

    @classmethod
    def from_packed(cls, b, sub=SUB, tile=1024, bias=None, alphas=None):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        if bias is not None:
            obj.bias = nn.Parameter(bias.clone())
        else:
            obj.register_parameter('bias', None)
        obj._init_from_buffers(b, sub, tile)
        return obj

    def _init_from_buffers(self, b, sub, tile):
        self.out_f = b['out_f']; self.in_f = b['in_f']
        self.sub = sub; self.tile = tile
        self._w_bf16 = None
        for k in ('emax', 'sign_packed', 'mant_packed', 'delta_packed',
                  'delta_ovf_idx', 'delta_ovf_val'):
            self.register_buffer(k, b[k])

    @property
    def weight(self):
        return torch.zeros(1, 1, dtype=torch.bfloat16, device=self.mant_packed.device)

    def _decode_tile(self, s, e):
        dev = self.mant_packed.device
        fs = s * self.in_f
        nv = (e - s) * self.in_f
        mant = unpack_bits_stream(self.mant_packed, fs * 7, nv * 7, 7).to(torch.int32)
        d = unpack_bits_stream(self.delta_packed, fs * 3, nv * 3, 3).to(torch.int32)
        sgn = unpack_bits_stream(self.sign_packed, fs * 1, nv * 1, 1).to(torch.int32)
        ovf = d == 7
        if ovf.any() and self.delta_ovf_idx.numel() > 0:
            pos = torch.arange(fs, fs + nv, device=dev)
            loc = torch.searchsorted(self.delta_ovf_idx.to(dev), pos)
            loc = loc.clamp(max=self.delta_ovf_idx.numel() - 1)
            d = torch.where(ovf, self.delta_ovf_val.to(dev).to(torch.int32)[loc], d)
        sub = torch.arange(fs, fs + nv, device=dev) // self.sub
        emax = self.emax.to(dev).to(torch.int32)[sub]
        expo = (emax - d) & 0xFF
        vbits = (sgn << 15) | (expo << 7) | (mant & 0x7F)
        return vbits.to(torch.int16).view(torch.bfloat16).reshape(e - s, self.in_f)

    def decode_to_bf16(self):
        chunks = []
        for s in range(0, self.out_f, self.tile):
            e = min(s + self.tile, self.out_f)
            chunks.append(self._decode_tile(s, e))
        self._w_bf16 = torch.cat(chunks, dim=0)
        return self._w_bf16

    def clear_bf16(self):
        self._w_bf16 = None

    def forward(self, x):
        if getattr(self, "_w_bf16", None) is not None:
            return torch.nn.functional.linear(x, self._w_bf16, self.bias)
        x_shape = x.shape
        x_flat = x.reshape(-1, self.in_f)
        outs = []
        for i in range(0, self.out_f, self.tile):
            end = min(i + self.tile, self.out_f)
            w_tile = self._decode_tile(i, end)
            outs.append(torch.matmul(x_flat, w_tile.t()))
            del w_tile
        out = torch.cat(outs, dim=-1).reshape(*x_shape[:-1], self.out_f)
        if self.bias is not None:
            out = out + self.bias
        return out

    def decoded_weight(self):
        return self._decode_tile(0, self.out_f).to(torch.float32)


# ============================================================
# 在线替换
# ============================================================
def replace_with_opqk(module, sub=SUB, tile=1024, count=None, min_el=50000, snr_target=None):
    if count is None:
        count = [0]
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.weight.numel() > min_el:
            try:
                setattr(module, name, OPQKLinear(child, sub=sub, tile=tile, snr_target=snr_target))
                count[0] += 1
            except Exception:
                pass
        else:
            replace_with_opqk(child, sub, tile, count, min_el, snr_target)
    return count[0]


def replace_with_bf16x(module, sub=SUB, tile=1024, count=None, min_el=50000):
    if count is None:
        count = [0]
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.weight.numel() > min_el:
            try:
                setattr(module, name, BF16XLinear(child, sub=sub, tile=tile))
                count[0] += 1
            except Exception:
                pass
        else:
            replace_with_bf16x(child, sub, tile, count, min_el)
    return count[0]


# ============================================================
# 自检
# ============================================================
def _snr(w, wq):
    mse = ((w.reshape(-1).float() - wq.reshape(-1)) ** 2).mean().item()
    var = w.float().var(unbiased=False).item()
    return 10 * torch.log10(torch.tensor(var / max(mse, 1e-15))).item()


def _exact(w_bf16, w_rec):
    eq = (w_rec.to(torch.bfloat16).view(torch.int16) == w_bf16.view(torch.int16))
    return eq.float().mean().item() * 100


def _bpw_opqk(b, avg_bits):
    return avg_bits + 0.125 + SCALE_BITS / SUB + 16.0 / SUPER + 3.0 / SUB


def _bpw_bf16x(b, N):
    sub = b['sub']
    mant_bits = N * 7
    del_bits = N * 3
    ovf = b['delta_ovf_idx'].numel()
    bits = mant_bits + del_bits + N + ovf * (32 + 8) + ((N + sub - 1) // sub) * 8
    return bits / N


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin = nn.Linear(5376, 28672, bias=True)          # cpu 上量化 (buffers 回 cpu)
    with torch.no_grad():
        lin.weight.mul_(0.13)
        idx = torch.randperm(lin.weight.numel())[:lin.weight.numel() // 200]
        lin.weight.reshape(-1)[idx] *= torch.randn(len(idx)) * 8

    print("== Mode1: OPQ-K (每子块α + 自适应int4-6) ==")
    t0 = time.time()
    qlin = OPQKLinear(lin, tile=1024)
    b = opqk_quantize(lin.weight.data, snr_target=None)
    wq = qlin.decoded_weight()
    avg_bits = b['bits'].float().mean().item()
    print("  quantize %.2fs | 平均bits=%.2f | BPW=%.2f | SNR=%.2f dB | bf16精确=%.2f%%"
          % (time.time() - t0, avg_bits, _bpw_opqk(b, avg_bits), _snr(lin.weight, wq), _exact(lin.weight.to(torch.bfloat16), wq)))

    print("== Mode1b: OPQ-K 固定 int6 ==")
    t0 = time.time()
    b6 = opqk_quantize(lin.weight.data, bits_opts=(6,), snr_target=None)
    q6 = OPQKLinear.from_packed(b6, tile=1024, bias=lin.bias.data)
    wq6 = q6.decoded_weight()
    print("  BPW=%.2f | SNR=%.2f dB | bf16精确=%.2f%%" % (_bpw_opqk(b6, 6), _snr(lin.weight, wq6), _exact(lin.weight.to(torch.bfloat16), wq6)))

    print("== Mode2: BF16X 精确无损 ==")
    t0 = time.time()
    bx = bf16x_quantize(lin.weight.data)
    qx = BF16XLinear.from_packed(bx, tile=1024, bias=lin.bias.data)
    wqx = qx.decoded_weight()
    print("  BPW=%.2f | bf16精确=%.2f%% (应=100)" % (_bpw_bf16x(bx, lin.weight.numel()), _exact(lin.weight.to(torch.bfloat16), wqx)))

    print("== forward (bf16, CUDA) ==")
    qlin.to(dev); q6.to(dev); qx.to(dev)
    lin_b = lin.to(dev).to(torch.bfloat16)
    x = torch.randn(4, 5376, device=dev, dtype=torch.bfloat16)
    y_orig = torch.nn.functional.linear(x, lin_b.weight, lin_b.bias).float()
    for tag, m in [("OPQ-K", qlin), ("OPQ-K int6", q6), ("BF16X", qx)]:
        y = m(x).float()
        rel = ((y_orig - y).norm() / y_orig.norm()).item()
        print("  %-10s forward relative error: %.4f" % (tag, rel))

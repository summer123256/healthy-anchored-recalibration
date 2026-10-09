"""
健康状态锚定的前向重校准（纯整数参考实现，与 STM32 端 tinycnn.c 逐位一致）
=====================================================================
思路：
  对每个卷积层 l 的每个通道 c，统计重量化之前的累加值 acc 的均值 μ 与标准差 σ。
  源域统计量（离线在电脑上算好，写进 Flash）：
      健康锚点  (μs_h, σs_h)：只用源域“正常”类窗口
      常规锚点  (μs_a, σs_a)：用源域全部类别窗口（= 常规 BN 重校准 / FORGE 类做法）
  现场只采集正常状态窗口，算出 (μt, σt)，按
      acc' = a·(acc − μt) + μs，   a = clip(σs/σt, 1/4, 4)
  对齐，并把它并入整数偏置与重量化乘数（权重不变）：
      b' = b − μt + round(μs / a)，  M' = a·M（重新规格化到 [2^30, 2^31)，同步调整移位 s）
  逐层顺序进行，每层用新的一批窗口（单片机 RAM 存不下多个窗口）。

全部运算为 Python 整数，舍入规则与 C 完全相同：
  div_round：四舍五入、远离零；isqrt：向下取整；a 用 Q16 定点数表示。
"""
import copy
import math

import numpy as np

from quant import conv_layer_indices, forward_acc

Q = 16
ONE = 1 << Q
AQ_MIN = 1 << (Q - 2)      # a >= 1/4
AQ_MAX = 1 << (Q + 2)      # a <= 4
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def div_round(a, b):
    """整数除法，四舍五入远离零；b > 0。与 C 端 tc_div_round64 一致。"""
    a, b = int(a), int(b)
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)


def stats_from_sums(s1, s2, n):
    """由 Σacc、Σacc²、个数 n 求 (μ, σ)，全部整数。"""
    mu = div_round(s1, n)
    var = div_round(s2, n) - mu * mu
    return mu, math.isqrt(max(var, 0))


def layer_sums(qlayers, xq, li, batch=128):
    """xq: (N, WIN) int8。返回该层每通道的 (Σacc, Σacc², 计数)，Python 大整数。"""
    s1 = s2 = None
    n = 0
    for i in range(0, len(xq), batch):
        acc = forward_acc(qlayers, xq[i:i + batch, None, :], li)       # (B, C, L) int64
        b1 = [int(v) for v in acc.sum(axis=(0, 2))]
        b2 = [int(v) for v in (acc * acc).sum(axis=(0, 2))]
        s1 = b1 if s1 is None else [a + b for a, b in zip(s1, b1)]
        s2 = b2 if s2 is None else [a + b for a, b in zip(s2, b2)]
        n += acc.shape[0] * acc.shape[2]
    return s1, s2, n


def source_stats(qlayers, xq, y, healthy_label=0):
    """离线计算源域统计量。xq: 源域训练窗口 (N, WIN) int8，y: 标签。
    返回 {层号: dict(mu_h, sig_h, mu_a, sig_a)}，每项为 int64 数组（每通道一个）。"""
    out = {}
    healthy = xq[y == healthy_label]
    if len(healthy) == 0:
        raise ValueError("源域中没有正常类样本")
    for li in conv_layer_indices(qlayers):
        res = {}
        for tag, data in (("h", healthy), ("a", xq)):
            s1, s2, n = layer_sums(qlayers, data, li)
            ms = [stats_from_sums(a, b, n) for a, b in zip(s1, s2)]
            res["mu_" + tag] = np.array([m for m, _ in ms], dtype=np.int64)
            res["sig_" + tag] = np.array([s for _, s in ms], dtype=np.int64)
        for k, v in res.items():
            if v.min() < INT32_MIN or v.max() > INT32_MAX:
                raise OverflowError(f"第 {li} 层统计量 {k} 超出 int32")
        out[li] = res
    return out


def update_channel(b, m, s, mu_t, sig_t, mu_s, sig_s, scale=True):
    """单个通道的整数更新。返回 (b', M', s')。与 C 端 tc_calib_apply 一致。"""
    if scale and sig_t > 0 and sig_s > 0:
        aq = div_round(sig_s * ONE, sig_t)
        aq = min(max(aq, AQ_MIN), AQ_MAX)
    else:
        aq = ONE
    b_new = b - mu_t + div_round(mu_s * ONE, aq)
    b_new = min(max(b_new, INT32_MIN), INT32_MAX)
    m64 = (m * aq + (ONE >> 1)) >> Q
    while m64 >= (1 << 31):
        m64 = (m64 + 1) >> 1
        s -= 1
    while m64 < (1 << 30):
        m64 <<= 1
        s += 1
    s = min(max(s, 1), 62)
    return b_new, m64, s


class WindowStream:
    """按顺序循环提供校准窗口，模拟单片机逐层采集新数据。"""

    def __init__(self, xq, shuffle_seed=None):
        self.x = np.asarray(xq, dtype=np.int8)
        if len(self.x) == 0:
            raise ValueError("校准窗口为空")
        if shuffle_seed is not None:
            self.x = self.x[np.random.default_rng(shuffle_seed).permutation(len(self.x))]
        self.pos = 0

    def take(self, n):
        idx = [(self.pos + i) % len(self.x) for i in range(n)]
        self.pos = (self.pos + n) % len(self.x)
        return self.x[idx]


def recalibrate(qlayers, stream, stats, anchor="healthy", scale=True, n_per_layer=32,
                n_layers=None):
    """在拷贝上执行逐层重校准并返回新模型。
    anchor: "healthy"（本文 M3）或 "all"（常规 M1/M2）
    scale : True 同时校正均值与尺度；False 只校正均值
    n_layers: 只校准前 n_layers 个卷积层（None 为全部）"""
    q = copy.deepcopy(qlayers)
    layers = conv_layer_indices(q)
    if n_layers is not None:
        layers = layers[:n_layers]
    tag = "h" if anchor == "healthy" else "a"
    for li in layers:
        L = q[li]
        s1, s2, n = layer_sums(q, stream.take(n_per_layer), li)
        mu_s = stats[li]["mu_" + tag]
        sig_s = stats[li]["sig_" + tag]
        b = [int(v) for v in L["bq"]]
        m = [int(v) for v in L["mult"]]
        sh = [int(v) for v in L["shift"]]
        for c in range(len(b)):
            mu_t, sig_t = stats_from_sums(s1[c], s2[c], n)
            b[c], m[c], sh[c] = update_channel(b[c], m[c], sh[c], mu_t, sig_t,
                                               int(mu_s[c]), int(sig_s[c]), scale)
        L["bq"] = np.array(b, dtype=np.int64)
        L["mult"] = np.array(m, dtype=np.int64)
        L["shift"] = np.array(sh, dtype=np.int64)
    return q


def mixed_stream(healthy_xq, fault_xq, frac_fault, seed=0):
    """构造被污染的校准流：frac_fault 比例的窗口来自故障类（用于消融实验）。"""
    rng = np.random.default_rng(seed)
    n = len(healthy_xq)
    k = int(round(n * frac_fault))
    x = np.array(healthy_xq, copy=True)
    if k and len(fault_xq):
        pos = rng.choice(n, size=k, replace=False)
        x[pos] = fault_xq[rng.choice(len(fault_xq), size=k, replace=True)]
    return x


# ---------------------------------------------------------------------------
# 对比方法（只在电脑上评估，单片机不实现）：与 M0–M3 使用同一个 int8 模型，
# 都锚定源域“全部类别”统计量，区别只在于“目标统计量”怎么估计。
# ---------------------------------------------------------------------------
def _per_window_stats(qlayers, xq, li):
    """每个窗口单独统计：返回 (均值, 方差)，形状 (N, C)，浮点。"""
    acc = forward_acc(qlayers, xq[:, None, :], li).astype(np.float64)     # (N, C, L)
    return acc.mean(axis=2), acc.var(axis=2)


def _recal_with(qlayers, stream, stats, n_per_layer, estimate):
    """通用逐层重校准：estimate(li, windows, mu_sA, var_sA) -> (mu_t, var_t)，浮点数组 (C,)"""
    q = copy.deepcopy(qlayers)
    for li in conv_layer_indices(q):
        L = q[li]
        mu_s = stats[li]["mu_a"].astype(np.float64)
        sig_s = stats[li]["sig_a"].astype(np.float64)
        mu_t, var_t = estimate(q, li, stream.take(n_per_layer), mu_s, sig_s ** 2)
        b = [int(v) for v in L["bq"]]
        m = [int(v) for v in L["mult"]]
        sh = [int(v) for v in L["shift"]]
        for c in range(len(b)):
            b[c], m[c], sh[c] = update_channel(b[c], m[c], sh[c], int(round(mu_t[c])),
                                               int(math.isqrt(max(int(round(var_t[c])), 0))),
                                               int(stats[li]["mu_a"][c]), int(stats[li]["sig_a"][c]), True)
        L["bq"] = np.array(b, dtype=np.int64)
        L["mult"] = np.array(m, dtype=np.int64)
        L["shift"] = np.array(sh, dtype=np.int64)
    return q


def recalibrate_alpha_bn(qlayers, stream, stats, alpha=0.5, n_per_layer=32):
    """B2：α-BN 思路（You 等, 2021）。目标统计量与源域统计量加权混合：
    μ = α·μ_t + (1−α)·μ_s，σ² = α·σ_t² + (1−α)·σ_s²，再对齐到源域全类别统计量。α=1 即常规重校准 M1。"""
    def est(q, li, xq, mu_s, var_s):
        mw, vw = _per_window_stats(q, xq, li)
        mu_t = mw.mean(0)
        var_t = (vw + mw ** 2).mean(0) - mu_t ** 2            # 合并所有窗口的总体方差
        return alpha * mu_t + (1 - alpha) * mu_s, alpha * var_t + (1 - alpha) * var_s
    return _recal_with(qlayers, stream, stats, n_per_layer, est)


def recalibrate_dua(qlayers, stream, stats, n_per_layer=32, m0=0.1, decay=0.94, m_min=0.005):
    """B1：DUA 思路（Mirza 等, 2022）的简化实现（不含原文的数据增强）。
    从源域统计量出发，逐个窗口用递减动量 m_k = max(m0·decay^k, m_min) 更新 (μ, σ²)，再对齐到源域全类别统计量。"""
    def est(q, li, xq, mu_s, var_s):
        mw, vw = _per_window_stats(q, xq, li)
        mu, var = mu_s.copy(), var_s.copy()
        for k in range(len(mw)):
            mom = max(m0 * decay ** k, m_min)
            mu = (1 - mom) * mu + mom * mw[k]
            var = (1 - mom) * var + mom * vw[k]
        return mu, var
    return _recal_with(qlayers, stream, stats, n_per_layer, est)

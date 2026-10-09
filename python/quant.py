"""
int8 量化与整数推理参考实现（纯 numpy，不依赖 PyTorch）
====================================================
量化方案（与 CMSIS-NN / TFLite Micro 的对称量化思路一致）：
  * 激活：int8 对称量化，零点为 0；网络输入尺度固定为 1/127
  * 权重：卷积与隐藏全连接层按输出通道量化；最后一层全连接按整层量化
          （整层统一尺度，保证 int32 输出直接取最大值即为分类结果）
  * 偏置：int32，尺度 = 输入尺度 × 权重尺度
  * 重量化：out = (acc × M + 2^(s-1)) >> s，M 为 [2^30, 2^31) 的 int32，s 为移位位数
BatchNorm 在量化前合并进卷积/全连接权重。

int_forward() 与 STM32 端 tinycnn.c 的运算逐位一致，
用来：①评估量化后精度；②生成单片机测试向量；③硬件在环时核对单片机输出。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from preprocess import div_round

INPUT_SCALE = 1.0 / 127.0


# ---------------------------------------------------------------------------
# 基本算子（float 与 int 共用）
# ---------------------------------------------------------------------------
def conv1d(x, w, stride, pad, groups):
    """x: (N, C, L)，w: (O, C/g, K)。返回 (N, O, Lout)，dtype 跟随 x。"""
    n, c, _ = x.shape
    o, cg, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = sliding_window_view(xp, k, axis=2)[:, :, ::stride, :]      # (N, C, Lout, K)
    lout = win.shape[2]
    win = win.reshape(n, groups, cg, lout, k)
    wg = w.reshape(groups, o // groups, cg, k).astype(x.dtype)
    out = np.einsum("ngclk,gock->ngol", win, wg, optimize=True)
    return out.reshape(n, o, lout)


def maxpool(x, k):
    n, c, l = x.shape
    lo = l // k
    return x[:, :, :lo * k].reshape(n, c, lo, k).max(axis=-1)


# ---------------------------------------------------------------------------
# BN 合并
# ---------------------------------------------------------------------------
def fold_bn(graph):
    out = []
    for L in graph:
        L = dict(L)
        if L["type"] in ("conv", "fc"):
            w = L["w"].astype(np.float64)
            b = np.zeros(w.shape[0]) if L["b"] is None else L["b"].astype(np.float64)
            bn = L.pop("bn", None)
            if bn is not None:
                s = bn["gamma"] / np.sqrt(bn["var"] + bn["eps"])
                w = w * s.reshape(-1, *([1] * (w.ndim - 1)))
                b = bn["beta"] + (b - bn["mean"]) * s
            L["w"], L["b"] = w, b
        out.append(L)
    return out


# ---------------------------------------------------------------------------
# 浮点前向（合并 BN 后），用于校准和对照
# ---------------------------------------------------------------------------
def float_forward(folded, x, collect=False):
    acts = []
    for L in folded:
        t = L["type"]
        if t == "conv":
            x = conv1d(x, L["w"], L["stride"], L["pad"], L["groups"]) + L["b"][None, :, None]
        elif t == "fc":
            x = x @ L["w"].T + L["b"]
        elif t == "maxpool":
            x = maxpool(x, L["k"])
        elif t == "gap":
            x = x.mean(axis=-1)
        elif t == "flatten":
            x = x.reshape(x.shape[0], -1)
        if t in ("conv", "fc") and L.get("relu"):
            x = np.maximum(x, 0)
        if t in ("conv", "fc"):
            acts.append(x)
    return (x, acts) if collect else x


# ---------------------------------------------------------------------------
# 量化
# ---------------------------------------------------------------------------
def quantize_multiplier(real):
    """real > 0 -> (M, s)，满足 real ≈ M / 2^s，M ∈ [2^30, 2^31)"""
    real = np.asarray(real, dtype=np.float64)
    mant, exp = np.frexp(real)                  # real = mant * 2^exp, mant ∈ [0.5, 1)
    m = np.rint(mant * (1 << 31)).astype(np.int64)
    exp = exp.astype(np.int64)
    over = m == (1 << 31)
    m[over] //= 2
    exp[over] += 1
    s = 31 - exp
    # 保证 1 <= s <= 62（移位范围）
    while np.any(s > 62):
        big = s > 62
        m[big] //= 2
        s[big] -= 1
    if np.any(s < 1):
        raise ValueError("重量化系数过大（>= 2^30），请检查校准数据")
    return m.astype(np.int64), s.astype(np.int64)


def quantize(graph, calib_x, act_percentile=None):
    """graph: models.export_graph() 的输出；calib_x: (N,1,WIN) 浮点输入（int8/127）。
    返回 qlayers 列表。"""
    folded = fold_bn(graph)
    _, acts = float_forward(folded, calib_x.astype(np.float64), collect=True)
    n_param = sum(1 for L in folded if L["type"] in ("conv", "fc"))
    q, s_in, ai, pi = [], INPUT_SCALE, 0, 0
    shape = (calib_x.shape[1], calib_x.shape[2])       # (C, L)
    for L in folded:
        t = L["type"]
        if t in ("conv", "fc"):
            pi += 1
            last = pi == n_param
            w, b = L["w"], L["b"]
            if last:
                assert t == "fc", "最后一层必须是全连接"
                sw = np.full(w.shape[0], max(np.abs(w).max(), 1e-12) / 127.0)
            else:
                sw = np.maximum(np.abs(w.reshape(w.shape[0], -1)).max(axis=1), 1e-12) / 127.0
            wq = np.clip(np.rint(w / sw.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)
            bq = np.rint(b / (s_in * sw)).astype(np.int64)
            ql = dict(type=t, wq=wq, bq=bq, relu=bool(L.get("relu")), last=last,
                      s_in=s_in, sw=sw, in_shape=shape)
            if t == "conv":
                ql.update(stride=L["stride"], pad=L["pad"], groups=L["groups"])
                lout = (shape[1] + 2 * L["pad"] - w.shape[2]) // L["stride"] + 1
                shape = (w.shape[0], lout)
            else:
                shape = (w.shape[0], 1)
            if not last:
                a = acts[ai]
                amax = np.percentile(np.abs(a), act_percentile) if act_percentile else np.abs(a).max()
                s_out = max(float(amax), 1e-8) / 127.0
                mult, shift = quantize_multiplier(s_in * sw / s_out)
                ql.update(mult=mult, shift=shift, s_out=s_out)
                s_in = s_out
            ql["out_shape"] = shape
            ai += 1
            q.append(ql)
        elif t == "maxpool":
            q.append(dict(type=t, k=L["k"], in_shape=shape))
            shape = (shape[0], shape[1] // L["k"])
            q[-1]["out_shape"] = shape
        elif t == "gap":
            q.append(dict(type=t, in_shape=shape, out_shape=(shape[0], 1)))
            shape = (shape[0], 1)
        elif t == "flatten":
            q.append(dict(type=t, in_shape=shape, out_shape=(shape[0] * shape[1], 1)))
            shape = (shape[0] * shape[1], 1)
    return q


# ---------------------------------------------------------------------------
# 整数推理（与 STM32 逐位一致）
# ---------------------------------------------------------------------------
def requant(acc, mult, shift, relu):
    """acc: (N, O, ...) int64；mult/shift: (O,)"""
    ext = (1, -1) + (1,) * (acc.ndim - 2)
    m = mult.reshape(ext)
    s = shift.reshape(ext)
    v = (acc * m + (np.int64(1) << (s - 1))) >> s
    return np.clip(v, 0 if relu else -127, 127)


def int_forward(qlayers, xq, batch=512):
    """xq: (N,1,WIN) int8。返回 int64 logits (N, n_classes)。"""
    outs = []
    for i in range(0, len(xq), batch):
        x = xq[i:i + batch].astype(np.int64)
        for L in qlayers:
            x = layer_step(L, x)
        outs.append(x.reshape(x.shape[0], -1))
    return np.concatenate(outs)


def conv_acc(L, x):
    """卷积层的 int32 累加值 acc = W·x + b（重量化之前）。x: (N, C, L) int64"""
    acc = conv1d(x, L["wq"].astype(np.int64), L["stride"], L["pad"], L["groups"])
    return acc + np.asarray(L["bq"], dtype=np.int64)[None, :, None]


def layer_step(L, x):
    """执行一层整数运算（与 C 端 tinycnn.c 一致）。"""
    t = L["type"]
    if t == "conv":
        acc = conv_acc(L, x)
        return acc if L["last"] else requant(acc, np.asarray(L["mult"], dtype=np.int64),
                                             np.asarray(L["shift"], dtype=np.int64), L["relu"])
    if t == "fc":
        x = x.reshape(x.shape[0], -1)
        acc = x @ L["wq"].astype(np.int64).T + np.asarray(L["bq"], dtype=np.int64)[None, :]
        return acc if L["last"] else requant(acc, np.asarray(L["mult"], dtype=np.int64),
                                             np.asarray(L["shift"], dtype=np.int64), L["relu"])
    if t == "maxpool":
        return maxpool(x, L["k"])
    if t == "gap":
        return div_round(x.sum(axis=-1), x.shape[-1])[:, :, None]
    if t == "flatten":
        return x.reshape(x.shape[0], -1, 1)
    raise ValueError(t)


def forward_acc(qlayers, xq, li):
    """前向计算到第 li 层（必须是卷积层），返回该层的累加值 acc (N, C, L) int64。"""
    x = np.asarray(xq, dtype=np.int64)
    for L in qlayers[:li]:
        x = layer_step(L, x)
    return conv_acc(qlayers[li], x)


def conv_layer_indices(qlayers):
    """可校准的层：除最后一层以外的全部卷积层。"""
    return [i for i, L in enumerate(qlayers) if L["type"] == "conv" and not L["last"]]


def memory_report(qlayers):
    """估算 STM32 上的存储占用（字节）。"""
    weights = sum(L["wq"].size for L in qlayers if "wq" in L)
    bias = sum(L["bq"].size * 4 for L in qlayers if "bq" in L)
    rq = sum(L["mult"].size * 4 + L["shift"].size for L in qlayers if "mult" in L)
    act = [int(np.prod(L["out_shape"])) for L in qlayers]
    inp = int(np.prod(qlayers[0]["in_shape"]))
    max_act = max(act + [inp])
    return dict(weight_bytes=int(weights), bias_bytes=int(bias), requant_bytes=int(rq),
                flash_model_bytes=int(weights + bias + rq),
                max_activation_bytes=int(max_act),
                ram_activation_bytes=int(2 * max_act))   # 乒乓缓冲区

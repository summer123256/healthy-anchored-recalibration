"""
命题 1–3 的数值验证（合成信号，纯 numpy，不需要 PyTorch、不需要下载数据）
======================================================================
目的：在“已知真实域偏移”的受控条件下，检查论文第 3 节三个命题及整数实现是否成立。
注意：这里用的是合成信号和一个“随机卷积特征 + 最小二乘分类层”的小网络，
      只用于验证理论与代码，不代表真实数据上的诊断准确率（真实结果见 experiments_tta.py）。

做法：
  1. 合成 3 类振动信号（正常 / 外圈型冲击 / 内圈型调制冲击），换算为 int16 计数
  2. 网络结构与本文 tiny 模型相同，卷积层随机初始化，最后一层全连接用岭回归拟合，然后 int8 量化
  3. 受控偏移：把第一个卷积层的累加值变为 acc_t = α_c·acc_s + β_c（逐通道）—— 即命题 1 的假设 A。
     实现方法：权重乘 α_c 后取整、偏置变为 α_c·b + β_c。
     α_c ∈ [0.5, 2] 连续取值时，权重取整使偏移只是“近似仿射”；α_c ∈ {1,2,3,4} 整数时为严格仿射
  4. 检查：
     V1（命题 1）有偏移时，M3 能否恢复源域模型的输出（logits 相对误差、与源模型预测一致率、准确率）
     V2（命题 2）没有偏移时，M1 是否反而改变了模型，M3 是否几乎不变
     V3（命题 3）校准数据混入比例 ε 的故障窗口时，第一层均值误差是否随 ε 线性增长
     V4 α 超出 [1/4, 4] 时的截断影响
输出：results/theory_check.json

用法：python check_theory.py            （约 1–2 分钟；5 个替代模型，报告均值 ± 标准差）
"""
import copy
import json

import numpy as np

from config import RESULT_DIR, WIN
from preprocess import int8_to_model_input, preprocess_int16, to_int16
from quant import conv_layer_indices, float_forward, fold_bn, forward_acc, int_forward, quantize
from recal import WindowStream, mixed_stream, recalibrate, source_stats

FS = 16000.0


def synth(rng, n, cls):
    """合成 1024 点窗口。cls 0 正常；1 外圈型（固定周期冲击）；2 内圈型（冲击幅值按转频调制）"""
    t = np.arange(WIN) / FS
    out = np.empty((n, WIN))
    for i in range(n):
        fr = rng.uniform(24, 26)                                   # 转频约 25 Hz
        x = sum(rng.uniform(300, 900) / h * np.sin(2 * np.pi * h * fr * t + rng.uniform(0, 6.3))
                for h in (1, 2, 3))
        x += rng.normal(0, 250, WIN)
        if cls:
            fchar = fr * (3.05 if cls == 1 else 4.95)              # 外圈 / 内圈特征频率（倍转频）
            period = FS / fchar
            imp = np.zeros(WIN)
            k = np.arange(rng.uniform(0, period), WIN, period).astype(int)
            amp = rng.uniform(2500, 4000) * (1 + (0.8 * np.cos(2 * np.pi * fr * k / FS) if cls == 2 else 0))
            imp[k] = amp
            fres = rng.uniform(2800, 3400)                         # 结构共振频率
            h = np.exp(-np.arange(80) / 12.0) * np.sin(2 * np.pi * fres * np.arange(80) / FS)
            x += np.convolve(imp, h)[:WIN]
        out[i] = x
    return out


def synth_healthy(rng, n):
    return preprocess_int16(to_int16(synth(rng, n, 0)))


def make_set(rng, n_per_class):
    X = np.concatenate([synth(rng, n_per_class, c) for c in range(3)])
    y = np.repeat(np.arange(3), n_per_class)
    return preprocess_int16(to_int16(X)), y


def build_model(rng, xq, y):
    """随机卷积（tiny 结构）+ 岭回归分类层，返回 int8 量化模型"""
    from selftest import random_graph
    g = random_graph("tiny", rng, n_cls=3)
    folded = fold_bn(g[:-1])                                       # 去掉最后的全连接，取 GAP 特征
    feat = float_forward(folded, int8_to_model_input(xq).astype(np.float64))
    F = np.hstack([feat, np.ones((len(feat), 1))])
    Y = np.eye(3)[y] * 2 - 1
    W = np.linalg.solve(F.T @ F + 1.0 * np.eye(F.shape[1]), F.T @ Y)   # 岭回归 λ=1（λ 过小时模型对微小扰动过于敏感）
    fc = g[-1]
    fc["w"] = W[:-1].T.copy()
    fc["b"] = W[-1].copy()
    return quantize(g, int8_to_model_input(xq[:600]))


def shift_first_layer(q, alpha, beta):
    """受控偏移：第一个可校准层 acc -> α·acc + β（权重乘 α 取整，偏置 α·b + β）"""
    qt = copy.deepcopy(q)
    li = conv_layer_indices(qt)[0]
    L = qt[li]
    L["wq"] = np.rint(L["wq"].astype(np.float64) * alpha[:, None, None]).astype(np.int64)
    L["bq"] = np.rint(np.asarray(L["bq"], dtype=np.float64) * alpha + beta).astype(np.int64)
    return qt


def compare(q_ref, q, xq, y):
    """与源域模型（无偏移、未校准）比较：logits 相对误差、预测一致率、准确率"""
    z0 = int_forward(q_ref, xq[:, None, :]).astype(np.float64)
    z = int_forward(q, xq[:, None, :]).astype(np.float64)
    return dict(logit_rel_err=float(np.linalg.norm(z - z0) / np.linalg.norm(z0)),
                agree_with_source=float((z.argmax(1) == z0.argmax(1)).mean()),
                acc=float((z.argmax(1) == y).mean()))


def run_one(seed):
    rng = np.random.default_rng(seed)
    xs_tr, ys_tr = make_set(rng, 400)
    xs_te, ys_te = make_set(rng, 150)
    xh_cal = synth_healthy(rng, 200)                               # 新的正常窗口（校准用，与训练集不重叠）
    q = build_model(rng, xs_tr, ys_tr)
    stats = source_stats(q, xs_tr, ys_tr)
    layers = conv_layer_indices(q)
    li0 = layers[0]
    C = len(stats[li0]["mu_h"])
    sig_h = stats[li0]["sig_h"].astype(np.float64)
    xs_h = xs_tr[ys_tr == 0]
    out = dict(seed=seed, source=compare(q, q, xs_te, ys_te))

    # V1 命题 1：受控仿射偏移（α∈[0.5,2]、β~N(0,σ²)，逐通道随机）
    r = np.random.default_rng(1000 + seed)
    alpha = np.exp(r.uniform(np.log(0.5), np.log(2.0), C))
    beta = r.normal(0, 1.0, C) * sig_h
    qt = shift_first_layer(q, alpha, beta)
    v1 = dict(M0=compare(q, qt, xs_te, ys_te))
    for name, anc in (("M1", "all"), ("M3", "healthy")):
        v1[name] = compare(q, recalibrate(qt, WindowStream(xh_cal, seed), stats, anc, True, 32), xs_te, ys_te)
    # 不含估计噪声的检查：校准窗口 = 计算源域统计量所用的同一批正常窗口，只校准第一层
    v1["M3_exact_stats_first_layer"] = compare(
        q, recalibrate(qt, WindowStream(xs_h), stats, "healthy", True, len(xs_h), 1), xs_te, ys_te)
    # 严格仿射偏移：α 取整数 {1,2,3,4}、β 取整数，此时 acc_t = α·acc_s + β 精确成立（无权重取整误差）
    a_int = r.integers(1, 5, C).astype(np.float64)
    qt_int = shift_first_layer(q, a_int, np.rint(beta))
    v1["M0_int_alpha"] = compare(q, qt_int, xs_te, ys_te)
    v1["M3_int_alpha_exact_stats_first_layer"] = compare(
        q, recalibrate(qt_int, WindowStream(xs_h), stats, "healthy", True, len(xs_h), 1), xs_te, ys_te)
    out["V1"] = v1

    # V2 命题 2：没有偏移
    v2 = {}
    for name, anc in (("M1", "all"), ("M3", "healthy")):
        v2[name] = compare(q, recalibrate(q, WindowStream(xh_cal, seed), stats, anc, True, 32), xs_te, ys_te)
    v2["M3_exact_stats"] = compare(q, recalibrate(q, WindowStream(xs_h), stats, "healthy", True, len(xs_h)),
                                   xs_te, ys_te)
    v2["M1_exact_stats"] = compare(q, recalibrate(q, WindowStream(xs_h), stats, "all", True, len(xs_h)),
                                   xs_te, ys_te)
    v2["anchor_gap_in_sigma_median_by_layer"] = [
        float(np.median(np.abs(stats[li]["mu_a"] - stats[li]["mu_h"]) / np.maximum(stats[li]["sig_h"], 1)))
        for li in layers]
    out["V2"] = v2

    # V3 命题 3：校准数据混入故障窗口（比例 ε）。在每个可校准层上测 |μ_t(ε) − μs_h| / σs_h 的通道中位数
    faults = xs_tr[ys_tr != 0]
    eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    lay = layers[len(layers) // 2]                                   # 取中间一层（第一层几乎不区分类别）
    big_h = xs_h
    errs = []
    for eps in eps_list:
        mixed = mixed_stream(big_h, faults, eps, seed=seed)
        mu_t = forward_acc(q, mixed[:, None, :], lay).astype(np.float64).mean(axis=(0, 2))
        errs.append(float(np.median(np.abs(mu_t - stats[lay]["mu_h"]) / np.maximum(stats[lay]["sig_h"], 1))))
    k, b0 = np.polyfit(eps_list, errs, 1)
    pred = k * np.array(eps_list) + b0
    r2 = 1 - np.sum((np.array(errs) - pred) ** 2) / max(np.sum((np.array(errs) - np.mean(errs)) ** 2), 1e-12)
    acc_eps = [compare(q, recalibrate(q, WindowStream(mixed_stream(xh_cal, faults, e, seed=seed), seed), stats,
                                      "healthy", True, 32), xs_te, ys_te)["acc"] for e in eps_list]
    out["V3"] = dict(layer=int(lay), eps=eps_list, mean_err_in_sigma=errs, linear_r2=float(r2),
                     slope=float(k), acc_M3=acc_eps)

    # V4 α = 6，超出截断上限 4
    qt6 = shift_first_layer(q, np.full(C, 6.0), np.zeros(C))
    out["V4"] = dict(M0=compare(q, qt6, xs_te, ys_te),
                     M3=compare(q, recalibrate(qt6, WindowStream(xs_h), stats, "healthy", True, len(xs_h), 1),
                                xs_te, ys_te))
    return out


V1_KEYS = ("M0", "M1", "M3", "M3_exact_stats_first_layer", "M0_int_alpha", "M3_int_alpha_exact_stats_first_layer")


def main():
    seeds = [0, 1, 2, 3, 4]                                          # 5 个不同的替代模型与数据
    runs = []
    for sd in seeds:
        o = run_one(sd)
        runs.append(o)
        print(f"模型 {sd}: 源域 {o['source']['acc']:.3f} | V1 M0 {o['V1']['M0']['acc']:.3f} M1 {o['V1']['M1']['acc']:.3f} "
              f"M3 {o['V1']['M3']['acc']:.3f} 一致率(精确统计) {o['V1']['M3_exact_stats_first_layer']['agree_with_source']:.3f} "
              f"(整数α) {o['V1']['M3_int_alpha_exact_stats_first_layer']['agree_with_source']:.3f} | "
              f"V2 M1 {o['V2']['M1']['acc']:.3f} M3 {o['V2']['M3']['acc']:.3f} | V3 R² {o['V3']['linear_r2']:.3f}")

    def ms(path):
        v = []
        for o in runs:
            x = o
            for p in path:
                x = x[p]
            v.append(x)
        return dict(mean=float(np.mean(v)), std=float(np.std(v)))

    summary = dict(
        note="合成信号 + 替代模型上的理论与代码验证，不是真实数据上的诊断结果",
        n_models=len(seeds),
        source_acc=ms(["source", "acc"]),
        V1_acc={m: ms(["V1", m, "acc"]) for m in V1_KEYS},
        V1_agree={m: ms(["V1", m, "agree_with_source"]) for m in V1_KEYS},
        V1_logit_err={m: ms(["V1", m, "logit_rel_err"]) for m in V1_KEYS},
        V2_acc={m: ms(["V2", m, "acc"]) for m in ("M1", "M3", "M1_exact_stats", "M3_exact_stats")},
        V2_agree={m: ms(["V2", m, "agree_with_source"]) for m in ("M1", "M3", "M1_exact_stats", "M3_exact_stats")},
        V3_r2=ms(["V3", "linear_r2"]),
        V3_err_by_eps=[ms(["V3", "mean_err_in_sigma", i]) for i in range(6)],
        V3_acc_by_eps=[ms(["V3", "acc_M3", i]) for i in range(6)],
        V4_acc={m: ms(["V4", m, "acc"]) for m in ("M0", "M3")},
        V4_agree={m: ms(["V4", m, "agree_with_source"]) for m in ("M0", "M3")},
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "theory_check.json").write_text(json.dumps(dict(summary=summary, runs=runs), indent=2,
                                                             ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"-> {RESULT_DIR / 'theory_check.json'}")


if __name__ == "__main__":
    main()

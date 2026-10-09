"""
片上无标签自适应的全部对比与消融实验（论文第 5 节的数据来源）
==========================================================
场景：
  C1  CWRU 跨传感器位置：驱动端 DE（0–3 HP）训练 → 风扇端 FE（0–3 HP）测试
  C2  CWRU 跨负载：驱动端 0 HP 训练 → 1 / 2 / 3 HP 分别测试
  P   帕德博恩 PU（真实损伤轴承）：N15_M07_F10 训练 → P1 换转速 / P2 换负载扭矩 / P3 换径向力
  F   风扇台（可选，需自采数据）：源域 S0 → F1–F4（缺少的场景自动跳过）
方法（M0–M3、B1、B2 为纯整数实现，与单片机逐位一致；M4、B3 需要 PyTorch）：
  M0 不自适应；M1 常规重校准（锚定源域全部类别，仅用正常数据）；
  M2 常规重校准（类别混合的无标签数据流，理想参考，现场通常拿不到）；
  M3 本文：健康状态锚定重校准（仅用正常数据）；M3L1 同 M3 但只校准第一个卷积层；M4 目标域有标签微调（上限，电脑上完成）；
  B1 DUA 思路的简化版（逐窗口递减动量更新统计量，不含原文的数据增强）；
  B2 α-BN（目标与源域统计量按 α 混合，默认 α=0.5）；
  B1、B2 与 M1/M3 用同一批“仅正常”窗口，锚点为源域全部类别（与原方法一致）；
  B3 TENT（熵最小化更新 BN 仿射参数，浮点、需反向传播，单片机上不可行）。默认只用正常数据（与本文设定公平）；
     --tent-data pool 改用含故障的混合无标签数据（现场拿不到，仅作参考，需在论文中单独标注）
实验：
  E0 无偏移检查：在源域上用源域正常窗口校准，再在源域测试集评估 M0/M1/M3 → _e0.csv
  E1 主结果（各目标域 × 各随机种子）→ _main.csv、_summary.csv
  E2 校准数据被故障窗口污染（比例 ε，M1/M3/M3L1）；E3 每层窗口数 N（M3/M3L1）；
  E4 消融（校准前 k 层 k=1..全部、只校正均值、B2 的 α）→ _ablation.csv
     （E2–E4 只在开发目标域（每个场景的第一个目标域）上做；正式运行时第一个种子做；
      --ablation-only 则每个种子都做，并对每个种子、每个目标域做 E5 → _e5_allseeds.json）
  E5 假设 A 检查：每个可校准层上，用正常类拟合每通道 (α, β)，预测故障类的目标域均值和标准差 → _e5.csv、_e5_summary.json

用法：
  python experiments_tta.py --scenario C1 --quick     # 快速跑通（1 个种子、少量轮数）
  python experiments_tta.py --scenario C1             # 正式（5 个种子、60 轮）
  python experiments_tta.py --scenario C2
  python experiments_tta.py --scenario P
  python experiments_tta.py --scenario C1 --ablation-only   # 5 个种子的消融 + E5（不覆盖主结果）
  python experiments_tta.py --scenario C1 --supplement      # E0 补 M3L1、B2 的 α 扫描 → _supp.csv
  （任何模式加 --use-ckpt：已保存的源域模型直接读取，省去训练时间）
  python experiments_tta.py --gate                    # 读取 C1、C2、P 结果，汇总 M3L1/M3 相对 M0 的提升及决策条件
可复现性：每个种子在建模型前固定全部随机源；如两次运行结果仍不同，可加 --threads 4 固定线程数
结果：results/tta_<场景>_main.csv、_summary.csv、_ablation.csv、_e0.csv、_e5.csv、_diag.csv、_source.json；tta_gate.json
"""
import argparse
import csv
import json
import time

import numpy as np
from sklearn.metrics import f1_score

from config import CKPT_DIR, CWRU_CLASSES, FAN_CLASSES, PU_CLASSES, PU_TARGETS, RESULT_DIR
from preprocess import int8_to_model_input, preprocess_int16, to_int16
from quant import conv_layer_indices, forward_acc, int_forward, quantize
from recal import (WindowStream, mixed_stream, recalibrate, recalibrate_alpha_bn, recalibrate_dua,
                   source_stats)

METHODS = ["M0", "M1", "M2", "M3", "M3L1", "M4", "B1", "B2", "B3"]


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def to_q(X):
    """浮点窗口（int16 计数单位）-> int8 窗口 (N, WIN)"""
    return preprocess_int16(to_int16(X))


def load_scenario(name):
    """返回 (源域 dict, 目标域列表, 类别名, 源域配置)。
    目标域：dict(name, healthy=X, pool=(X, y), test=(X, y))"""
    if name in ("C1", "C2"):
        from data_cwru import build_cwru, build_cwru_target
        if name == "C1":
            src_cfg = dict(dataset="cwru", channel="DE", train_loads=[0, 1, 2, 3])
            src = build_cwru((0, 1, 2, 3), "DE")
            tgt_specs = [("C1", (0, 1, 2, 3), "FE")]
        else:
            src_cfg = dict(dataset="cwru", channel="DE", train_loads=[0])
            src = build_cwru((0,), "DE")
            tgt_specs = [(f"C2-{h}HP", (h,), "DE") for h in (1, 2, 3)]
        targets = []
        for tname, loads, ch in tgt_specs:
            t = build_cwru_target(loads, ch, synchronous=(name == "C1"))
            Xc, yc, _ = t["calib"]
            Xt, yt, _ = t["test"]
            targets.append(dict(name=tname, healthy=Xc[yc == 0], pool=(Xc, yc), test=(Xt, yt)))
        return src, targets, CWRU_CLASSES, src_cfg
    if name == "P":
        from data_pu import build_pu_source, build_pu_target
        src = build_pu_source()
        targets = []
        for tname in PU_TARGETS:
            t = build_pu_target(tname)
            Xc, yc, _ = t["calib"]
            Xt, yt, _ = t["test"]
            targets.append(dict(name=tname, healthy=Xc[yc == 0], pool=(Xc, yc), test=(Xt, yt)))
        return src, targets, PU_CLASSES, dict(dataset="pu")
    if name == "F":
        from data_fan import build_fan_source, build_fan_target
        src = build_fan_source()
        targets = []
        for dom in ("F1", "F2", "F3", "F4"):
            try:
                t = build_fan_target(dom)
            except (FileNotFoundError, ValueError) as e:
                print(f"  跳过 {dom}：{e}")
                continue
            targets.append(dict(name=dom, healthy=t["calib_healthy"][0],
                                pool=t["pool"][:2], test=t["test"][:2]))
        if not targets:
            raise FileNotFoundError("没有任何风扇目标域数据（F1–F4）")
        return src, targets, FAN_CLASSES, dict(dataset="fan")
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 训练、量化、评估
# ---------------------------------------------------------------------------
def set_determinism(seed, threads=None):
    """固定所有随机源，使同一种子在同一台电脑上重复运行得到相同结果。
    注意：模型初始权重在 build_model 时就已抽取，所以必须在建模型之前设种子。"""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if threads:
        torch.set_num_threads(threads)


def train_source(src, n_classes, epochs, seed, model_name="tiny"):
    from models import build_model
    from train import train_model
    set_determinism(seed)
    m = build_model(model_name, n_classes)
    train_model(m, src["train"], src["val"], epochs=epochs, seed=seed, verbose=False)
    return m


def get_source_model(a, src, classes, epochs, seed):
    """训练源域模型；每个种子的模型都存到 checkpoints/tta_<场景>_s<种子>_e<轮数>.pt。
    加 --use-ckpt 时，若该文件已存在则直接读取（训练是确定的，读取与重新训练结果完全相同），可省去训练时间。"""
    import torch
    from models import build_model
    path = CKPT_DIR / f"tta_{a.scenario}_s{seed}_e{epochs}.pt"
    if a.use_ckpt and path.exists():
        print(f"种子 {seed}：读取已保存的源域模型 {path.name}")
        set_determinism(seed)
        m = build_model("tiny", len(classes))
        m.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        m.eval()
        return m
    print(f"种子 {seed}：训练源域模型（{epochs} 轮）…")
    m = train_source(src, len(classes), epochs, seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=m.state_dict(), model="tiny", n_classes=len(classes), classes=list(classes),
                    seed=seed, epochs=epochs), path)
    return m


SUPP_ALPHAS = (0.05, 0.1, 0.2, 0.3)


def run_supplement(q, stats, src, targets, seed, n_per_layer):
    """补充实验：(1) E0 无偏移检查补上 M3L1 和不同 α 的 B2；
    (2) 在每个目标域上评估不同 α 的 B2（α-BN），用于“在开发目标域上为 B2 选 α，再在留出目标域上报告”的公平比较。"""
    out = []
    Xv, yv, _ = src["val"]
    healthy = to_q(Xv[yv == 0])
    xq_test, y = to_q(src["test"][0]), src["test"][1]
    acc, f1 = evaluate_q(recalibrate(q, WindowStream(healthy, seed), stats, "healthy", True, n_per_layer, 1),
                         xq_test, y)
    out.append(dict(target="E0", method="M3L1", acc=acc, macro_f1=f1))
    for al in SUPP_ALPHAS:
        acc, f1 = evaluate_q(recalibrate_alpha_bn(q, WindowStream(healthy, seed), stats, al, n_per_layer), xq_test, y)
        out.append(dict(target="E0", method=f"B2_a{al}", acc=acc, macro_f1=f1))
    for tgt in targets:
        xt, yt, h = to_q(tgt["test"][0]), tgt["test"][1], to_q(tgt["healthy"])
        for al in SUPP_ALPHAS:
            acc, f1 = evaluate_q(recalibrate_alpha_bn(q, WindowStream(h, seed), stats, al, n_per_layer), xt, yt)
            out.append(dict(target=tgt["name"], method=f"B2_a{al}", acc=acc, macro_f1=f1))
    return out


def quantize_model(model, X_calib, n=1000, seed=0):
    from models import export_graph
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_calib), size=min(n, len(X_calib)), replace=False)
    return quantize(export_graph(model), int8_to_model_input(to_q(X_calib[idx])))


def finetune_m4(model, pool, epochs, seed):
    """M4：在目标域有标签数据上微调（上限参考），返回量化模型。"""
    import copy
    from train import train_model
    m = copy.deepcopy(model)
    train_model(m, (pool[0], pool[1], pool[1]), (pool[0], pool[1], pool[1]),
                epochs=epochs, lr=3e-4, seed=seed, verbose=False)
    return quantize_model(m, pool[0], seed=seed)


def tent_adapt(model, X, epochs=1, lr=1e-3, batch=64, seed=0):
    """B3：TENT（Wang 等, ICLR 2021）。只更新 BN 的 γ、β，目标为预测熵最小；BN 使用批统计量。
    适应结束后，用同一批数据重新累计 BN 的均值/方差（累积平均），再量化成 int8 模型以便与其他方法同样评估。"""
    import copy
    import torch
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)]
    for p in m.parameters():
        p.requires_grad_(False)
    params = []
    for bn in bns:
        bn.weight.requires_grad_(True)
        bn.bias.requires_grad_(True)
        params += [bn.weight, bn.bias]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    m.train()
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(int8_to_model_input(to_q(X[idx])))
            p = torch.softmax(m(xb), 1)
            loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():                        # 固定 BN 统计量（部署需要）
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
        for i in range(0, len(X), 256):
            m(torch.from_numpy(int8_to_model_input(to_q(X[i:i + 256]))))
    m.eval()
    return m


def evaluate_q(q, xq_test, y):
    pred = int_forward(q, xq_test[:, None, :]).argmax(1)
    return float((pred == y).mean()), float(f1_score(y, pred, average="macro"))


def run_methods(q, stats, tgt, seed, n_per_layer=32, scale=True, n_layers=None, q_m4=None,
                q_b3=None, alpha=0.5):
    """M0–M3、M3L1、B1、B2（纯整数，与单片机一致）以及可选的 M4、B3。
    返回 ({方法: (acc, f1)}, 诊断量 dict)"""
    xq_test = to_q(tgt["test"][0])
    y = tgt["test"][1]
    healthy = to_q(tgt["healthy"])
    pool = to_q(tgt["pool"][0])
    res = {"M0": evaluate_q(q, xq_test, y)}
    kw = dict(scale=scale, n_per_layer=n_per_layer, n_layers=n_layers)
    res["M1"] = evaluate_q(recalibrate(q, WindowStream(healthy, seed), stats, "all", **kw), xq_test, y)
    res["M2"] = evaluate_q(recalibrate(q, WindowStream(pool, seed), stats, "all", **kw), xq_test, y)
    q3 = recalibrate(q, WindowStream(healthy, seed), stats, "healthy", **kw)
    res["M3"] = evaluate_q(q3, xq_test, y)
    # M3L1：只校准第一个卷积层（假设 A 最可能在浅层成立；见论文 3.4 节对假设 A 的说明）
    q3l1 = recalibrate(q, WindowStream(healthy, seed), stats, "healthy", scale=scale,
                       n_per_layer=n_per_layer, n_layers=1)
    res["M3L1"] = evaluate_q(q3l1, xq_test, y)
    # 诊断量（不用任何标签）：已知为正常的校准窗口中，被模型判为“正常”的比例
    hr = lambda qq: float((int_forward(qq, healthy[:, None, :]).argmax(1) == 0).mean())  # noqa: E731
    diag = dict(healthy_rate_M0=hr(q), healthy_rate_M3=hr(q3), healthy_rate_M3L1=hr(q3l1),
                n_healthy_calib=int(len(healthy)))
    if q_m4 is not None:
        res["M4"] = evaluate_q(q_m4, xq_test, y)
    res["B1"] = evaluate_q(recalibrate_dua(q, WindowStream(healthy, seed), stats, n_per_layer), xq_test, y)
    res["B2"] = evaluate_q(recalibrate_alpha_bn(q, WindowStream(healthy, seed), stats, alpha, n_per_layer),
                           xq_test, y)
    if q_b3 is not None:
        res["B3"] = evaluate_q(q_b3, xq_test, y)
    return res, diag


def run_ablation(q, stats, tgt, seed, n_default):
    xq_test = to_q(tgt["test"][0])
    y = tgt["test"][1]
    healthy = to_q(tgt["healthy"])
    pool_x = to_q(tgt["pool"][0])
    faults = pool_x[tgt["pool"][1] != 0]
    rows = []

    def add(group, setting, qq):
        acc, f1 = evaluate_q(qq, xq_test, y)
        rows.append(dict(group=group, setting=setting, acc=acc, macro_f1=f1))

    n_all = len([L for L in q if L["type"] == "conv" and not L["last"]])
    hs = lambda: WindowStream(healthy, seed)  # noqa: E731
    for nl in range(1, n_all + 1):                      # E4：校准前 k 层（k=1 即 M3L1，k=全部 即 M3）
        add("layers", f"first_{nl}", recalibrate(q, hs(), stats, "healthy", True, n_default, nl))
    for nl, tag in ((None, "M3"), (1, "M3L1")):         # E4：只校正均值 vs 均值+尺度
        add(f"scale_{tag}", "mean_only", recalibrate(q, hs(), stats, "healthy", False, n_default, nl))
        add(f"scale_{tag}", "mean_and_scale", recalibrate(q, hs(), stats, "healthy", True, n_default, nl))
    for n in (4, 8, 16, 32, 64):                        # E3：每层窗口数 N
        add("windows", f"N={n}", recalibrate(q, hs(), stats, "healthy", True, n))
        add("windows_M3L1", f"N={n}", recalibrate(q, hs(), stats, "healthy", True, n, 1))
    for p in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):          # E2：同一污染数据流上比较 M1、M3、M3L1
        mixed = mixed_stream(healthy, faults, p, seed)
        add("contamination_M3", f"eps={p}", recalibrate(q, WindowStream(mixed, seed), stats, "healthy",
                                                         True, n_default))
        add("contamination_M3L1", f"eps={p}", recalibrate(q, WindowStream(mixed, seed), stats, "healthy",
                                                           True, n_default, 1))
        add("contamination_M1", f"eps={p}", recalibrate(q, WindowStream(mixed, seed), stats, "all",
                                                         True, n_default))
    for al in (0.1, 0.3, 0.5, 0.7, 0.9):                # B2 对 α 的敏感性
        add("alpha_bn", f"alpha={al}", recalibrate_alpha_bn(q, hs(), stats, al, n_default))
    return rows


def run_e0(q, stats, src, seed, n_per_layer):
    """E0：没有域偏移时的检查。用源域验证集的正常窗口校准，在源域测试集上评估。
    理论（命题 1、2）：M3 应与 M0 基本相同；M1 会因“正常 ≠ 全部类别”的偏差而下降。"""
    Xv, yv, _ = src["val"]
    healthy = to_q(Xv[yv == 0])
    xq_test, y = to_q(src["test"][0]), src["test"][1]
    out = {"M0": evaluate_q(q, xq_test, y)}
    for m, anc in (("M1", "all"), ("M3", "healthy")):
        out[m] = evaluate_q(recalibrate(q, WindowStream(healthy, seed), stats, anc, True, n_per_layer),
                            xq_test, y)
    return out


def _class_means(q, li, xq, y, n_cls, max_per_class=400, seed=0):
    """第 li 层累加值每类、每通道的均值与标准差，返回 (n_cls, C) 两个数组（缺类为 nan）。"""
    rng = np.random.default_rng(seed)
    mu, sd = None, None
    for k in range(n_cls):
        idx = np.flatnonzero(y == k)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, size=min(max_per_class, len(idx)), replace=False)
        acc = forward_acc(q, xq[idx][:, None, :], li).astype(np.float64)
        if mu is None:
            mu = np.full((n_cls, acc.shape[1]), np.nan)
            sd = np.full((n_cls, acc.shape[1]), np.nan)
        mu[k] = acc.mean(axis=(0, 2))
        sd[k] = acc.std(axis=(0, 2))
    return mu, sd


def _r2(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    return float(1 - np.sum((t - p) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-12))


def run_e5(q, src, tgt, n_cls, seed, classes, stats=None):
    """E5：检查假设 A（目标域 = 源域的逐通道仿射变换，且各类别共用同一变换）。
    对每个可校准层（未校准的 M0 模型），用正常类拟合每通道 α_c = σt/σs、β_c = μt − α_c·μs，
    再预测每个故障类在目标域的均值（α·μs + β）和标准差（α·σs），与实测值比较。
    注意：预处理对每个窗口去均值，第一层各类别的均值几乎都等于偏置，均值 R² 在第一层几乎必然接近 1，
    不能单独作为证据；判断主要看各层的“尺度”（标准差）预测和深层的均值预测。
    同时输出命题 2 中常规锚点的偏差项 (μs_all − μs_healthy)/σs_healthy。"""
    layers = conv_layer_indices(q)
    xs, ys = to_q(src["train"][0]), src["train"][1]
    xt, yt = to_q(tgt["pool"][0]), tgt["pool"][1]
    rows = []
    r2_mean, r2_std, err_mean, err_std = [], [], [], []
    for li in layers:
        mus, sds = _class_means(q, li, xs, ys, n_cls, seed=seed)
        mut, sdt = _class_means(q, li, xt, yt, n_cls, seed=seed)
        if mus is None or mut is None or np.isnan(mus[0]).any() or np.isnan(mut[0]).any():
            continue
        alpha = sdt[0] / np.maximum(sds[0], 1e-9)
        beta = mut[0] - alpha * mus[0]
        pm, tm, ps, ts, scale = [], [], [], [], []
        for k in range(1, n_cls):
            if np.isnan(mus[k]).any() or np.isnan(mut[k]).any():
                continue
            pred_mu, pred_sd = alpha * mus[k] + beta, alpha * sds[k]
            for c in range(len(pred_mu)):
                rows.append(dict(target=tgt["name"], layer=int(li), cls=classes[k], channel=c,
                                 mu_target_true=mut[k, c], mu_target_pred=pred_mu[c],
                                 sd_target_true=sdt[k, c], sd_target_pred=pred_sd[c],
                                 sigma_target_healthy=sdt[0, c]))
            pm.append(pred_mu); tm.append(mut[k]); ps.append(pred_sd); ts.append(sdt[k])
            scale.append(np.maximum(sdt[0], 1e-9))
        if not pm:
            continue
        pm, tm, ps, ts, scale = map(np.concatenate, (pm, tm, ps, ts, scale))
        r2_mean.append(_r2(tm, pm))
        r2_std.append(_r2(ts, ps))
        err_mean.append(float(np.median(np.abs(tm - pm) / scale)))
        err_std.append(float(np.median(np.abs(ts - ps) / np.maximum(ts, 1e-9))))
    if not rows:
        return [], {}
    # 命题 2 偏差项（第一层）：源域全部类别均值相对健康均值的偏离（以健康标准差为单位）
    li0 = layers[0]
    mus0, sds0 = _class_means(q, li0, xs, ys, n_cls, seed=seed)
    idx = np.random.default_rng(seed).choice(len(ys), size=min(3000, len(ys)), replace=False)
    acc_all = forward_acc(q, xs[idx][:, None, :], li0).astype(np.float64)
    bias = np.abs(acc_all.mean(axis=(0, 2)) - mus0[0]) / np.maximum(sds0[0], 1e-9)
    deep = slice(1, None)
    summ = dict(target=tgt["name"], layers=[int(l) for l in layers],
                r2_mean_by_layer=r2_mean, r2_std_by_layer=r2_std,
                median_err_mean_in_sigma_by_layer=err_mean,
                median_rel_err_std_by_layer=err_std,
                r2_std_median=float(np.median(r2_std)),
                r2_mean_deep_median=float(np.median(r2_mean[deep])) if len(r2_mean) > 1 else float("nan"),
                # 兼容旧字段：r2 取各层尺度 R² 的中位数（不再用第一层均值 R²，后者几乎必然接近 1）
                r2=float(np.median(r2_std)),
                median_abs_err_in_sigma=float(np.median(err_mean)),
                m1_anchor_bias_median_in_sigma=float(np.median(bias)),
                m1_anchor_bias_max_in_sigma=float(bias.max()))
    if stats is not None:     # 命题 2 的偏差项在每个可校准层上的大小（第一层往往很小，偏差主要在深层）
        summ["anchor_gap_median_in_sigma_by_layer"] = [
            float(np.median(np.abs(stats[l]["mu_a"] - stats[l]["mu_h"]).astype(np.float64)
                            / np.maximum(stats[l]["sig_h"], 1))) for l in layers]
    return rows, summ


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def summarize(rows):
    out = []
    keys = sorted({(r["target"], r["method"]) for r in rows}, key=lambda k: (k[0], METHODS.index(k[1])))
    for tgt, m in keys:
        a = np.array([r["acc"] for r in rows if r["target"] == tgt and r["method"] == m])
        f = np.array([r["macro_f1"] for r in rows if r["target"] == tgt and r["method"] == m])
        out.append(dict(target=tgt, method=m, acc_mean=round(a.mean() * 100, 2),
                        acc_std=round(a.std() * 100, 2), f1_mean=round(f.mean() * 100, 2),
                        f1_std=round(f.std() * 100, 2), n_seeds=len(a)))
    return out


def gate():
    """决策门槛（改版）：主方法预先声明为 M3L1（只校准第一层），同时报告 M3（全部层）。
    开发目标域：每个场景的第一个目标域（C1、C2-1HP、P1），消融只在这些目标上做；
    留出目标域：其余（C2-2HP、C2-3HP、P2、P3），只在最后报告，不用来调参。"""
    res, src_acc = {}, {}
    for sc in ("C1", "C2", "P"):
        p = RESULT_DIR / f"tta_{sc}_summary.csv"
        if not p.exists():
            print(f"缺少 {p}，请先运行 --scenario {sc}")
            continue
        with open(p, encoding="utf-8") as f:
            res[sc] = list(csv.DictReader(f))
        src_acc[sc] = json.loads((RESULT_DIR / f"tta_{sc}_source.json").read_text())["source_acc_mean"]
    if not res:
        return
    dev = {"C1", "C2-1HP", "P1"}
    table = {}
    for sc, rows in res.items():
        for r in rows:
            table.setdefault(r["target"], {})[r["method"]] = float(r["acc_mean"])
    lines, d_l1, d_m3, held_l1, l1_vs_m1 = [], [], [], [], []
    for t, m in table.items():
        dl1, dm3 = m["M3L1"] - m["M0"], m["M3"] - m["M0"]
        d_l1.append(dl1)
        d_m3.append(dm3)
        l1_vs_m1.append(m["M3L1"] > m["M1"])
        if t not in dev:
            held_l1.append(dl1)
        lines.append(dict(target=t, role="开发" if t in dev else "留出", M0=m["M0"], M1=m["M1"],
                          M3=m["M3"], M3L1=m["M3L1"], M3L1_minus_M0=round(dl1, 2), M3_minus_M0=round(dm3, 2)))
    cond = {"1_M3L1不出现大幅退化(每个目标域都>=M0-5)": bool(min(d_l1) >= -5),
            "2_M3L1在多数目标域上优于M1": bool(sum(l1_vs_m1) > len(l1_vs_m1) / 2),
            "3_M3L1平均比M0高>=3个百分点": bool(np.mean(d_l1) >= 3),
            "4_留出目标域上M3L1平均不低于M0": bool(len(held_l1) == 0 or np.mean(held_l1) >= 0)}
    out = dict(source_acc=src_acc, per_target=lines, mean_M3L1_minus_M0=round(float(np.mean(d_l1)), 2),
               mean_M3_minus_M0=round(float(np.mean(d_m3)), 2),
               heldout_mean_M3L1_minus_M0=round(float(np.mean(held_l1)), 2) if held_l1 else None,
               conditions=cond, passed=all(cond.values()))
    (RESULT_DIR / "tta_gate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("决策：" + ("全部条件通过" if out["passed"] else "有条件未通过，按《补充注意事项》中的诚实写法调整论文定位"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["C1", "C2", "P", "F"])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="默认 0 1 2 3 4")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ft-epochs", type=int, default=None, help="M4 微调轮数")
    ap.add_argument("--n-per-layer", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5, help="B2 α-BN 的混合系数")
    ap.add_argument("--tent-data", default="healthy", choices=["pool", "healthy"],
                    help="B3 用哪些目标域数据：healthy=只用正常数据（与本文设定公平，默认）；pool=混合含故障数据（仅作参考）")
    ap.add_argument("--threads", type=int, default=None, help="固定 PyTorch 线程数（可选，进一步保证可复现）")
    ap.add_argument("--no-m4", action="store_true")
    ap.add_argument("--no-b3", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--ablation-all", action="store_true", help="在所有目标域上做 E2–E4")
    ap.add_argument("--use-ckpt", action="store_true", help="若已保存同种子、同轮数的源域模型则直接读取，不重新训练")
    ap.add_argument("--supplement", action="store_true",
                    help="补充实验：E0 加 M3L1、各目标域上 B2 的 α 扫描（0.05/0.1/0.2/0.3）→ _supp.csv，不覆盖其他结果")
    ap.add_argument("--ablation-only", action="store_true",
                    help="只做 E2–E4 消融（每个种子都做）和 E5（每个种子、每个目标域），不重跑主结果；"
                         "输出 _ablation.csv 和 _e5_allseeds.json，不覆盖 _main.csv")
    a = ap.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if a.gate:
        gate()
        return
    if not a.scenario:
        ap.error("请指定 --scenario C1 / C2 / P / F，或 --gate")
    seeds = a.seeds or ([0] if a.quick else [0, 1, 2, 3, 4])
    epochs = a.epochs or (3 if a.quick else 60)
    ft_epochs = a.ft_epochs or (2 if a.quick else 20)

    t0 = time.time()
    set_determinism(0, a.threads)
    print(f"读取场景 {a.scenario} 的数据…")
    src, targets, classes, src_cfg = load_scenario(a.scenario)
    print(f"  源域训练 {len(src['train'][1])}，目标域 {[t['name'] for t in targets]}")
    rows, abl, src_accs, e0, e5, e5s, diags = [], [], [], [], [], [], []
    xq_src_test = to_q(src["test"][0])
    for seed in seeds:
        model = get_source_model(a, src, classes, epochs, seed)
        q = quantize_model(model, src["train"][0], seed=seed)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(src["train"][1]), size=min(3000, len(src["train"][1])), replace=False)
        stats = source_stats(q, to_q(src["train"][0][idx]), src["train"][1][idx], healthy_label=0)
        s_acc, _ = evaluate_q(q, xq_src_test, src["test"][1])
        src_accs.append(s_acc)
        if a.supplement:
            print(f"  源域测试准确率（int8）：{s_acc:.4f}")
            sup = run_supplement(q, stats, src, targets, seed, a.n_per_layer)
            for r in sup:
                abl.append(dict(scenario=a.scenario, seed=seed, **r))
            print("  " + "  ".join(f"{r['target']}:{r['method']}={r['acc']:.4f}" for r in sup))
            continue
        if a.ablation_only:
            print(f"  源域测试准确率（int8）：{s_acc:.4f}")
            for ti, tgt in enumerate(targets):
                if ti == 0 or a.ablation_all:
                    print(f"  {tgt['name']} E2–E4 消融…")
                    for r in run_ablation(q, stats, tgt, seed, a.n_per_layer):
                        abl.append(dict(scenario=a.scenario, target=tgt["name"], seed=seed, **r))
                _, s5 = run_e5(q, src, tgt, len(classes), seed, classes, stats)
                if s5:
                    e5s.append(dict(seed=seed, **s5))
                    f3 = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"  # noqa: E731
                    print(f"  {tgt['name']} E5 尺度 R² {f3(s5['r2_std_by_layer'])}")
            continue
        print(f"  源域测试准确率（int8）：{s_acc:.4f}")
        for m, (acc, f1) in run_e0(q, stats, src, seed, a.n_per_layer).items():
            e0.append(dict(scenario=a.scenario, seed=seed, method=m, acc=acc, macro_f1=f1))
        print("  E0 无偏移：" + "  ".join(f"{r['method']}={r['acc']:.4f}" for r in e0 if r["seed"] == seed))
        if seed == seeds[0]:
            import torch
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), model="tiny", n_classes=len(classes),
                            classes=list(classes), args=src_cfg),
                       CKPT_DIR / f"tta_{a.scenario}_s{seed}.pt")
        for ti, tgt in enumerate(targets):
            q_m4 = None if a.no_m4 else finetune_m4(model, tgt["pool"], ft_epochs, seed)
            q_b3 = None
            if not a.no_b3:
                Xb3 = tgt["pool"][0] if a.tent_data == "pool" else tgt["healthy"]
                q_b3 = quantize_model(tent_adapt(model, Xb3, seed=seed), Xb3, seed=seed)
            res, diag = run_methods(q, stats, tgt, seed, a.n_per_layer, q_m4=q_m4, q_b3=q_b3, alpha=a.alpha)
            diags.append(dict(scenario=a.scenario, target=tgt["name"], seed=seed, **diag))
            for m, (acc, f1) in res.items():
                rows.append(dict(scenario=a.scenario, target=tgt["name"], seed=seed, method=m,
                                 acc=acc, macro_f1=f1))
            print("  " + tgt["name"] + "  " + "  ".join(f"{m}={v[0]:.4f}" for m, v in res.items()))
            print(f"    正常校准窗口被判为正常的比例：M0={diag['healthy_rate_M0']:.3f}  "
                  f"M3={diag['healthy_rate_M3']:.3f}  M3L1={diag['healthy_rate_M3L1']:.3f}"
                  f"（共 {diag['n_healthy_calib']} 个窗口，不用标签）")
            if seed == seeds[0]:
                r5, s5 = run_e5(q, src, tgt, len(classes), seed, classes, stats)
                e5 += r5
                if s5:
                    e5s.append(s5)
                    f3 = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"  # noqa: E731
                    print(f"  E5 假设 A（各可校准层）：尺度 R² {f3(s5['r2_std_by_layer'])}；"
                          f"均值 R² {f3(s5['r2_mean_by_layer'])}（第一层均值 R² 无参考意义）")
            if not a.no_ablation and seed == seeds[0] and (ti == 0 or a.ablation_all):
                print("  E2–E4 消融…")
                for r in run_ablation(q, stats, tgt, seed, a.n_per_layer):
                    abl.append(dict(scenario=a.scenario, target=tgt["name"], seed=seed, **r))
    if a.supplement:
        write_csv(RESULT_DIR / f"tta_{a.scenario}_supp.csv", abl)
        print(f"  -> {RESULT_DIR / f'tta_{a.scenario}_supp.csv'}")
        print(f"完成，用时 {time.time() - t0:.0f} 秒")
        return
    if a.ablation_only:
        write_csv(RESULT_DIR / f"tta_{a.scenario}_ablation.csv", abl)
        (RESULT_DIR / f"tta_{a.scenario}_e5_allseeds.json").write_text(
            json.dumps(e5s, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {RESULT_DIR / f'tta_{a.scenario}_ablation.csv'}")
        print(f"  -> {RESULT_DIR / f'tta_{a.scenario}_e5_allseeds.json'}")
        print(f"完成，用时 {time.time() - t0:.0f} 秒")
        return
    write_csv(RESULT_DIR / f"tta_{a.scenario}_main.csv", rows)
    write_csv(RESULT_DIR / f"tta_{a.scenario}_summary.csv", summarize(rows))
    write_csv(RESULT_DIR / f"tta_{a.scenario}_ablation.csv", abl)
    write_csv(RESULT_DIR / f"tta_{a.scenario}_e0.csv", e0)
    write_csv(RESULT_DIR / f"tta_{a.scenario}_e5.csv", e5)
    write_csv(RESULT_DIR / f"tta_{a.scenario}_diag.csv", diags)
    (RESULT_DIR / f"tta_{a.scenario}_e5_summary.json").write_text(json.dumps(e5s, indent=2, ensure_ascii=False))
    (RESULT_DIR / f"tta_{a.scenario}_source.json").write_text(json.dumps(
        dict(source_acc_mean=float(np.mean(src_accs) * 100), source_acc_std=float(np.std(src_accs) * 100),
             seeds=seeds, epochs=epochs, n_per_layer=a.n_per_layer, alpha=a.alpha,
             tent_data=a.tent_data), indent=2))
    print(f"完成，用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()

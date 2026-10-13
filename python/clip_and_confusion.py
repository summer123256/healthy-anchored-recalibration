"""
补充统计：M3L1 第一层增益截断次数 + 混淆矩阵（不重新训练，只读取已保存的源域模型）
=====================================================================================
1) 增益截断：M3L1 第一层每个通道的增益 a = σs/σt 被限制在 [1/4, 4]。统计每个目标域、每个种子、
   16 个通道中 a 碰到下限或上限的次数（另外也统计无偏移检查 E0）。
2) 混淆矩阵：M0、M1、M3L1 在每个目标域上的混淆矩阵（5 个种子累加），并对指定目标域画图。

做法与 experiments_tta.py 完全相同（同一模型、同一量化、同一锚点、同一批校准窗口），
并逐项自检：
  - 用这里记录的增益重算的第一层参数，必须与 recalibrate() 得到的完全一致；
  - M3L1、M0、M1 的准确率与 results/tta_<场景>_main.csv 中的数值一致（若该文件存在）。

用法（在 python 文件夹下，激活原来的虚拟环境）：
  python clip_and_confusion.py                    # C1、C2、P、J 四个场景，种子 0–4
  python clip_and_confusion.py --scenarios C2 P   # 只做部分场景
需要 checkpoints/tta_<场景>_s<种子>_e60.pt（之前用 --use-ckpt 跑过的模型）；缺少的会跳过，不会重新训练。
结果：results/clip_confusion/ 下的
  clip_counts.csv（每目标域、每种子）、clip_summary.txt（汇总，发给我这个文件即可）、
  confusion_<目标域>_<方法>.csv（5 个种子累加的计数）、confusion_<目标域>.png（行归一化百分比）
"""
import argparse
import csv
import json
import time
from types import SimpleNamespace

import numpy as np

from config import CKPT_DIR, RESULT_DIR
from experiments_tta import (evaluate_q, get_source_model, load_scenario, quantize_model, set_determinism,
                             to_q)
from quant import conv_layer_indices, int_forward
from recal import (AQ_MAX, AQ_MIN, ONE, WindowStream, div_round, layer_sums, recalibrate, source_stats,
                   stats_from_sums, update_channel)

OUT = RESULT_DIR / "clip_confusion"
METHODS = ("M0", "M1", "M3L1")


def first_layer_gains(q, stats, healthy_xq, seed, n_per_layer):
    """按 recalibrate(anchor='healthy', n_layers=1) 的同一流程，记录第一层每个通道的增益。
    返回 (记录列表, 用这些增益算出的新参数 (b, M, s))。"""
    li = conv_layer_indices(q)[0]
    L = q[li]
    s1, s2, n = layer_sums(q, WindowStream(healthy_xq, seed).take(n_per_layer), li)
    mu_s, sig_s = stats[li]["mu_h"], stats[li]["sig_h"]
    recs, new_params = [], []
    for c in range(len(s1)):
        mu_t, sig_t = stats_from_sums(s1[c], s2[c], n)
        ss = int(sig_s[c])
        if sig_t > 0 and ss > 0:
            aq_raw = div_round(ss * ONE, sig_t)
            kind = "low" if aq_raw < AQ_MIN else ("high" if aq_raw > AQ_MAX else "none")
        else:
            aq_raw = None
            kind = "zero_sigma"
        recs.append(dict(channel=c, a_raw=(aq_raw / ONE if aq_raw is not None else None), clip=kind))
        new_params.append(update_channel(int(L["bq"][c]), int(L["mult"][c]), int(L["shift"][c]),
                                         mu_t, sig_t, int(mu_s[c]), ss, True))
    return recs, new_params


def check_params(q3l1, new_params):
    li = conv_layer_indices(q3l1)[0]
    L = q3l1[li]
    for c, (b, m, s) in enumerate(new_params):
        if (int(L["bq"][c]), int(L["mult"][c]), int(L["shift"][c])) != (b, m, s):
            raise AssertionError(f"自检失败：第 {c} 通道的参数与 recalibrate() 不一致")


def confusion(y, pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y.astype(int), pred.astype(int)), 1)
    return cm


def load_main(scenario):
    path = RESULT_DIR / f"tta_{scenario}_main.csv"
    if not path.exists():
        return None
    ref = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[(r["target"], int(r["seed"]), r["method"])] = float(r["acc"])
    return ref


def plot_confusions(name, cms, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(classes)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.4 * len(METHODS) + 0.6, 3.4 if k <= 4 else 4.2))
    for ax, m in zip(axes, METHODS):
        cm = cms[m].astype(np.float64)
        pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        acc = 100 * np.trace(cm) / cm.sum()
        ax.set_title(f"{m} ({acc:.1f}%)", fontsize=9)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(classes, rotation=90 if k > 4 else 0, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        if m == METHODS[0]:
            ax.set_ylabel("True", fontsize=8)
        if k <= 4:
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if pct[i, j] > 60 else "black")
    fig.suptitle(f"{name}: confusion matrices summed over seeds (row-normalised, %)", fontsize=9)
    fig.tight_layout()
    path = OUT / f"confusion_{name}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["C1", "C2", "P", "J"], choices=["C1", "C2", "P", "J"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-layer", type=int, default=32)
    ap.add_argument("--plot", nargs="+", default=["P1", "C2-3HP", "C1", "J2"], help="画混淆矩阵图的目标域")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    clip_rows, gain_rows, summary, checks = [], [], {}, []
    for sc in a.scenarios:
        missing = [s for s in a.seeds if not (CKPT_DIR / f"tta_{sc}_s{s}_e{a.epochs}.pt").exists()]
        if missing:
            print(f"场景 {sc}：缺少种子 {missing} 的模型文件（checkpoints/tta_{sc}_s*_e{a.epochs}.pt），跳过该场景")
            continue
        set_determinism(0)
        print(f"读取场景 {sc} 的数据…")
        src, targets, classes, _ = load_scenario(sc)
        ref = load_main(sc)
        cms = {(t["name"], m): np.zeros((len(classes), len(classes)), dtype=np.int64)
               for t in targets for m in METHODS}
        for seed in a.seeds:
            # 与 experiments_tta.main() 相同的顺序：读模型 → 量化 → 锚点
            model = get_source_model(SimpleNamespace(scenario=sc, use_ckpt=True), src, classes, a.epochs, seed)
            q = quantize_model(model, src["train"][0], seed=seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(src["train"][1]), size=min(3000, len(src["train"][1])), replace=False)
            stats = source_stats(q, to_q(src["train"][0][idx]), src["train"][1][idx], healthy_label=0)
            # E0：源域验证段的正常窗口
            Xv, yv, _ = src["val"]
            evals = [("E0 (no shift)", to_q(Xv[yv == 0]), None)] + [(t["name"], to_q(t["healthy"]), t) for t in targets]
            for name, healthy, tgt in evals:
                recs, new_params = first_layer_gains(q, stats, healthy, seed, a.n_per_layer)
                q3l1 = recalibrate(q, WindowStream(healthy, seed), stats, "healthy", scale=True,
                                   n_per_layer=a.n_per_layer, n_layers=1)
                check_params(q3l1, new_params)
                n_low = sum(r["clip"] == "low" for r in recs)
                n_high = sum(r["clip"] == "high" for r in recs)
                n_zero = sum(r["clip"] == "zero_sigma" for r in recs)
                araw = [r["a_raw"] for r in recs if r["a_raw"] is not None]
                clip_rows.append(dict(scenario=sc, target=name, seed=seed, channels=len(recs), clipped_low=n_low,
                                      clipped_high=n_high, zero_sigma=n_zero,
                                      a_min=round(min(araw), 4) if araw else "", a_max=round(max(araw), 4) if araw else ""))
                for r in recs:
                    gain_rows.append(dict(scenario=sc, target=name, seed=seed, **r))
                if tgt is None:
                    continue
                xt, yt = to_q(tgt["test"][0]), tgt["test"][1]
                qm1 = recalibrate(q, WindowStream(healthy, seed), stats, "all", scale=True,
                                  n_per_layer=a.n_per_layer, n_layers=None)
                for m, qq in (("M0", q), ("M1", qm1), ("M3L1", q3l1)):
                    pred = int_forward(qq, xt[:, None, :]).argmax(1)
                    acc = float((pred == yt).mean())
                    cms[(name, m)] += confusion(yt, pred, len(classes))
                    if ref is not None and (name, seed, m) in ref:
                        ok = abs(ref[(name, seed, m)] - acc) < 1e-9
                        checks.append(dict(scenario=sc, target=name, seed=seed, method=m, acc=acc,
                                           acc_main_csv=ref[(name, seed, m)], match=ok))
                        if not ok:
                            print(f"  注意：{name} 种子 {seed} {m} 准确率 {acc:.6f} 与 main.csv 的 "
                                  f"{ref[(name, seed, m)]:.6f} 不一致")
                rng_txt = f"{min(araw):.3f} - {max(araw):.3f}" if araw else "-"
                print(f"  种子 {seed} {name:14s} 截断：下限 {n_low}，上限 {n_high}，sigma=0 {n_zero}；a 范围 {rng_txt}")
        for t in targets:
            for m in METHODS:
                cm = cms[(t["name"], m)]
                with open(OUT / f"confusion_{t['name']}_{m}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["true\\pred"] + list(classes))
                    for i, row in enumerate(cm):
                        w.writerow([classes[i]] + [int(v) for v in row])
            if t["name"] in a.plot:
                print(f"  -> {plot_confusions(t['name'], {m: cms[(t['name'], m)] for m in METHODS}, list(classes))}")
            summary[t["name"]] = {m: dict(acc=round(float(100 * np.trace(cms[(t['name'], m)]) / cms[(t['name'], m)].sum()), 2),
                                          per_class_recall=[round(float(100 * cms[(t['name'], m)][i, i] /
                                                                  max(cms[(t['name'], m)][i].sum(), 1)), 1)
                                                            for i in range(len(classes))])
                                  for m in METHODS}
            summary[t["name"]]["classes"] = list(classes)
    if not clip_rows:
        print("没有可处理的场景。请确认 checkpoints 文件夹里有 tta_<场景>_s<种子>_e60.pt。")
        return
    for path, rows in ((OUT / "clip_counts.csv", clip_rows), (OUT / "gains_per_channel.csv", gain_rows),
                       (OUT / "accuracy_check.csv", checks)):
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    # 汇总
    lines = ["M3L1 第一层增益截断统计（每个目标域 × 种子 × 16 个通道）", ""]
    groups = {}
    for r in clip_rows:
        key = "E0" if r["target"].startswith("E0") else ("JNU" if r["scenario"] == "J" else "main")
        g = groups.setdefault(key, dict(n=0, low=0, high=0, zero=0, amin=9e9, amax=0))
        g["n"] += r["channels"]
        g["low"] += r["clipped_low"]
        g["high"] += r["clipped_high"]
        g["zero"] += r["zero_sigma"]
        if r["a_min"] != "":
            g["amin"] = min(g["amin"], r["a_min"])
            g["amax"] = max(g["amax"], r["a_max"])
    label = {"main": "主目标域（C1、C2、P）", "JNU": "JNU 的 J1、J2", "E0": "无偏移检查 E0"}
    for key in ("main", "JNU", "E0"):
        if key in groups:
            g = groups[key]
            lines.append(f"{label[key]}：共 {g['n']} 个通道估计，碰到下限 1/4 {g['low']} 次，碰到上限 4 {g['high']} 次，"
                         f"sigma 为 0 {g['zero']} 次；a 范围 {g['amin']:.3f} - {g['amax']:.3f}")
    lines += ["", f"每个目标域（{len(a.seeds)} 个种子合计）："]
    per_t = {}
    for r in clip_rows:
        p = per_t.setdefault((r["scenario"], r["target"]), [0, 0, 0, 9e9, 0])
        p[0] += r["channels"]
        p[1] += r["clipped_low"] + r["clipped_high"]
        p[2] += r["zero_sigma"]
        if r["a_min"] != "":
            p[3] = min(p[3], r["a_min"])
            p[4] = max(p[4], r["a_max"])
    for (sc, t), p in per_t.items():
        lines.append(f"  {sc:3s} {t:14s} 通道 {p[0]:4d}  截断 {p[1]:3d}  sigma=0 {p[2]:2d}  a 范围 {p[3]:.3f} - {p[4]:.3f}")
    if checks:
        n_ok = sum(c["match"] for c in checks)
        lines += ["", f"准确率自检（与 results/tta_<场景>_main.csv 比较）：{n_ok}/{len(checks)} 一致"]
    else:
        lines += ["", "准确率自检：没有找到 results/tta_<场景>_main.csv，未比较"]
    lines += ["", f"混淆矩阵（{len(a.seeds)} 个种子累加；准确率 %，以及各类别召回率 %）："]
    for name, d in summary.items():
        lines.append(f"  {name}  类别 {d['classes']}")
        for m in METHODS:
            lines.append(f"    {m:5s} 准确率 {d[m]['acc']:6.2f}  各类召回率 {d[m]['per_class_recall']}")
    lines.append(f"\n用时 {time.time() - t0:.0f} 秒")
    text = "\n".join(lines)
    (OUT / "clip_summary.txt").write_text(text, encoding="utf-8")
    (OUT / "confusion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + text)
    print(f"\n结果在 {OUT}")


if __name__ == "__main__":
    main()

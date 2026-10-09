"""
显著性检验（论文表 7）
============================
把 C1、C2、P 三个场景的 results/tta_<场景>_main.csv 合并，按“目标域 × 随机种子”配对，
对参考方法（默认 M3L1，可用 --ref M3 改）与其他每个方法做 Wilcoxon 符号秩检验（双侧，正态近似含并列校正），
并给出 Holm 校正后的 p 值与平均差值；另外按目标域输出胜/负次数（每个目标域只有 5 个配对，
单个目标域上 Wilcoxon 最小 p=0.0625，所以只报告胜负，不做单目标域检验）。

用法：python stats_test.py                  （默认场景 C1 C2 P，缺少的场景跳过）
      python stats_test.py --scenarios C1 C2
      python stats_test.py --ref M3
      python stats_test.py --held-out      （只用留出目标域，论文表 7 下半部分）
若有 results/tta_<场景>_supp.csv，自动加入 B2*（开发集选定 α=0.1 的 α-BN），与论文表 7 一致（Holm 校正含 9 个比较）。
输出：results/stats_wilcoxon.csv
说明：配对数太少（< 6）时 Wilcoxon 检验无法得到 p < 0.05，结果仅供参考，论文中要如实写明配对数。
"""
import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon

from config import RESULT_DIR


def load(scenarios):
    rows = []
    for sc in scenarios:
        p = RESULT_DIR / f"tta_{sc}_main.csv"
        if not p.exists():
            print(f"跳过 {sc}：没有 {p}")
            continue
        with open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
        sp = RESULT_DIR / f"tta_{sc}_supp.csv"       # 补充实验：开发集选定 α=0.1 的 α-BN，记作 B2*
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                rows += [dict(r, method="B2*") for r in csv.DictReader(f)
                         if r["method"] == "B2_a0.1" and r["target"] != "E0"]
    return rows


HELD = {"C2-2HP", "C2-3HP", "P2", "P3"}          # 留出目标域（开发目标域为 C1、C2-1HP、P1）


def wilcoxon_p(d):
    """双侧 Wilcoxon 符号秩检验，统一用正态近似（含并列校正），零差值按 zsplit 处理。
    统一用正态近似是为了避免 scipy 在“有无并列”时自动切换精确法/近似法，使不同比较之间口径一致。"""
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d, zero_method="zsplit", method="approx").pvalue)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["C1", "C2", "P"])
    ap.add_argument("--metric", default="acc", choices=["acc", "macro_f1"])
    ap.add_argument("--ref", default="M3L1", help="参考方法（默认 M3L1）")
    ap.add_argument("--held-out", action="store_true", help="只用 4 个留出目标域（C2-2HP、C2-3HP、P2、P3）做检验")
    a = ap.parse_args()
    rows = load(a.scenarios)
    if not rows:
        raise SystemExit("没有任何结果文件，请先运行 experiments_tta.py")
    table = {}
    for r in rows:
        if a.held_out and r["target"] not in HELD:
            continue
        table.setdefault((r["target"], r["seed"]), {})[r["method"]] = float(r[a.metric])
    ref = a.ref
    others = sorted({r["method"] for r in rows} - {ref})
    out, ps = [], []
    for m in others:
        pairs = [(v[ref], v[m]) for v in table.values() if ref in v and m in v]
        if len(pairs) < 2:
            continue
        x = np.array(pairs)
        d = np.round((x[:, 0] - x[:, 1]) * 100, 6)      # 取整到 1e-6 个百分点，使真正相等的差值被识别为并列
        p = wilcoxon_p(d)
        ps.append(p)
        out.append(dict(comparison=f"{ref} vs {m}", n_pairs=len(pairs), mean_diff_pp=round(float(d.mean()), 2),
                        median_diff_pp=round(float(np.median(d)), 2), wins=int((d > 0).sum()),
                        ties=int((d == 0).sum()), losses=int((d < 0).sum()), p_raw=p))
    for r, pa in zip(out, holm(np.array(ps))):
        r["p_holm"] = float(pa)
    tag = "_heldout" if a.held_out else ""
    path = RESULT_DIR / f"stats_wilcoxon_{ref}{tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['comparison']:12s} 配对 {r['n_pairs']:3d}  平均差 {r['mean_diff_pp']:+6.2f} 个百分点  "
              f"胜/平/负 {r['wins']}/{r['ties']}/{r['losses']}  p={r['p_raw']:.4g}  Holm p={r['p_holm']:.4g}")
    print(f"-> {path}")
    # 按目标域的胜/负（参考方法 vs M0、M1）
    per = []
    targets = sorted({t for t, _ in table})
    for t in targets:
        vs = [v for (tt, _), v in table.items() if tt == t and ref in v]
        row = dict(target=t, n_seeds=len(vs))
        for m in ("M0", "M1", "M3", "M2"):
            if m == ref or not all(m in v for v in vs):
                continue
            d = np.array([(v[ref] - v[m]) * 100 for v in vs])
            row[f"mean_diff_vs_{m}"] = round(float(d.mean()), 2)
            row[f"wins_vs_{m}"] = f"{int((d > 0).sum())}/{len(d)}"
        per.append(row)
    path2 = RESULT_DIR / f"stats_per_target_{ref}{tag}.csv"
    keys = sorted({k for r in per for k in r}, key=lambda k: (k != "target", k != "n_seeds", k))
    with open(path2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per)
    print(f"\n按目标域（{ref} 减其他方法，单位：百分点；胜=种子数）：")
    for r in per:
        print("  " + r["target"].ljust(8) + "  ".join(
            f"vs {m}: {r[f'mean_diff_vs_{m}']:+6.2f}（胜 {r[f'wins_vs_{m}']}）"
            for m in ("M0", "M1", "M3", "M2") if f"mean_diff_vs_{m}" in r))
    print(f"-> {path2}")


if __name__ == "__main__":
    main()

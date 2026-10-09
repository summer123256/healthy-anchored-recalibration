"""
公平比较：为 B2（α-BN）在开发目标域上选 α，再在留出目标域上与 M3L1 比较
==========================================================================
规则（与本文选择“只校准第一层”时相同）：
  开发目标域 = C1、C2-1HP、P1（每个场景的第一个目标域）
  留出目标域 = C2-2HP、C2-3HP、P2、P3
  1) 在开发目标域上，取 5 个种子平均准确率最高的 α（候选 0.05/0.1/0.2/0.3/0.5）作为 B2*；
  2) 在留出目标域上比较 M3L1 与 B2*（20 个配对的 Wilcoxon 符号秩检验，正态近似含并列校正，与 stats_test.py 一致）；同时给出全部 7 个目标域的结果。
输入：results/tta_<场景>_main.csv 与 results/tta_<场景>_supp.csv（先运行 experiments_tta.py --supplement）
输出：results/b2_tuned_report.json，并在屏幕上打印表格
用法：python b2_tuned_report.py
"""
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

from config import RESULT_DIR

DEV = ["C1", "C2-1HP", "P1"]
HELD = ["C2-2HP", "C2-3HP", "P2", "P3"]
ALPHAS = {"B2_a0.05": 0.05, "B2_a0.1": 0.1, "B2_a0.2": 0.2, "B2_a0.3": 0.3, "B2": 0.5}


def main():
    T = {}
    for sc in ("C1", "C2", "P"):
        for name in (f"tta_{sc}_main.csv", f"tta_{sc}_supp.csv"):
            p = RESULT_DIR / name
            if not p.exists():
                raise SystemExit(f"缺少 {p}")
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    T[(r["target"], int(r["seed"]), r["method"])] = float(r["acc"]) * 100
    seeds = sorted({s for (_, s, _) in T})
    mean = lambda t, m: float(np.mean([T[(t, s, m)] for s in seeds]))  # noqa: E731
    dev_score = {m: np.mean([mean(t, m) for t in DEV]) for m in ALPHAS}
    best = max(dev_score, key=dev_score.get)
    print("开发目标域上 B2 各 α 的平均准确率：" +
          "  ".join(f"α={ALPHAS[m]}: {v:.2f}" for m, v in dev_score.items()))
    print(f"选定 B2* = α={ALPHAS[best]}（开发目标域平均最高）\n")
    methods = ["M0", "M3L1", best]
    print("目标域    用途  " + "".join(f"{m:>10s}" for m in methods))
    per = []
    for t in DEV + HELD:
        row = dict(target=t, role="开发" if t in DEV else "留出", **{m: round(mean(t, m), 2) for m in methods})
        per.append(row)
        print(f"{t:8s}  {row['role']}  " + "".join(f"{row[m]:10.2f}" for m in methods))

    def test(ts):
        x = np.array([T[(t, s, "M3L1")] for t in ts for s in seeds])
        y = np.array([T[(t, s, best)] for t in ts for s in seeds])
        d = np.round(x - y, 6)
        p = float(wilcoxon(d, zero_method="zsplit", method="approx").pvalue) if np.any(d != 0) else 1.0
        return dict(n_pairs=len(d), mean_diff_pp=round(float(d.mean()), 2), wins=int((d > 0).sum()),
                    losses=int((d < 0).sum()), p=p)
    res = dict(alpha_selected=ALPHAS[best], dev_scores={str(ALPHAS[m]): round(float(v), 2) for m, v in dev_score.items()},
               per_target=per, M3L1_vs_B2star_heldout=test(HELD), M3L1_vs_B2star_all=test(DEV + HELD))
    for k in ("M3L1_vs_B2star_heldout", "M3L1_vs_B2star_all"):
        r = res[k]
        print(f"\n{k}: 配对 {r['n_pairs']}  平均差 {r['mean_diff_pp']:+.2f} 个百分点  胜/负 {r['wins']}/{r['losses']}  p={r['p']:.4g}")
    out = RESULT_DIR / "b2_tuned_report.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

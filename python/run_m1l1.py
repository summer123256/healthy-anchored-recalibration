"""
补充对照实验 M1L1：只校准第一个卷积层，但锚定源域“全部类别”统计量
=====================================================================
M1L1 与 M3L1 只差一处：第一层使用的源域锚点（全部类别 vs 正常类）。
层数、校准窗口（同一个 WindowStream、同一个种子）、整数舍入和裁剪规则完全相同。
用于回答：常规重校准如果也只动第一层，是否同样能避免崩溃？

同时输出：
  - M3L1 的重算值，并与已有 tta_<场景>_main.csv 中的 M3L1 逐项比较（应完全相同，用于确认设置一致）
  - E0 无偏移检查：源域验证集正常窗口校准、源域测试集评估（M0 / M1L1 / M3L1）
  - 第一层每个通道 σ_s^A / σ_s^H 的比值（两种锚点在第一层的差别主要在尺度上）

用法（需已有原始数据和 checkpoints，与 experiments_tta.py 相同）：
  python run_m1l1.py --use-ckpt                 # C1、C2、P 三个场景，5 个种子
  python run_m1l1.py --use-ckpt --scenarios P   # 只跑某个场景
结果：results/tta_m1l1.csv、results/tta_m1l1_e0.csv、results/tta_m1l1_check.json
"""
import argparse
import csv
import json
import time
from types import SimpleNamespace

import numpy as np

from config import RESULT_DIR
from experiments_tta import (evaluate_q, get_source_model, load_scenario, quantize_model,
                             set_determinism, to_q)
from quant import conv_layer_indices
from recal import WindowStream, recalibrate, source_stats


def read_main_m3l1(scenario):
    """读取已有主结果中的 M3L1 准确率：{(target, seed): acc}"""
    p = RESULT_DIR / f"tta_{scenario}_main.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] == "M3L1":
                out[(r["target"], int(r["seed"]))] = float(r["acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["C1", "C2", "P"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-layer", type=int, default=32)
    ap.add_argument("--use-ckpt", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, e0, check = [], [], {}
    for sc in a.scenarios:
        set_determinism(0, a.threads)
        print(f"读取场景 {sc} 的数据…")
        src, targets, classes, _ = load_scenario(sc)
        old = read_main_m3l1(sc)
        diffs, ratios = [], []
        xq_src_test, y_src_test = to_q(src["test"][0]), src["test"][1]
        Xv, yv, _ = src["val"]
        src_healthy = to_q(Xv[yv == 0])
        for seed in a.seeds:
            args = SimpleNamespace(scenario=sc, use_ckpt=a.use_ckpt)
            model = get_source_model(args, src, classes, a.epochs, seed)
            q = quantize_model(model, src["train"][0], seed=seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(src["train"][1]), size=min(3000, len(src["train"][1])), replace=False)
            stats = source_stats(q, to_q(src["train"][0][idx]), src["train"][1][idx], healthy_label=0)
            l1 = conv_layer_indices(q)[0]
            r = stats[l1]["sig_a"].astype(float) / np.maximum(stats[l1]["sig_h"].astype(float), 1)
            ratios += r.tolist()
            # E0：无偏移
            e0.append(dict(scenario=sc, seed=seed, method="M0", acc=evaluate_q(q, xq_src_test, y_src_test)[0]))
            for m, anc in (("M1L1", "all"), ("M3L1", "healthy")):
                qq = recalibrate(q, WindowStream(src_healthy, seed), stats, anc, True, a.n_per_layer, 1)
                e0.append(dict(scenario=sc, seed=seed, method=m, acc=evaluate_q(qq, xq_src_test, y_src_test)[0]))
            # 主对照：各目标域
            for tgt in targets:
                xq_test, y = to_q(tgt["test"][0]), tgt["test"][1]
                healthy = to_q(tgt["healthy"])
                res = {"M0": evaluate_q(q, xq_test, y)}
                for m, anc in (("M1L1", "all"), ("M3L1", "healthy")):
                    qq = recalibrate(q, WindowStream(healthy, seed), stats, anc, True, a.n_per_layer, 1)
                    res[m] = evaluate_q(qq, xq_test, y)
                for m, (acc, f1) in res.items():
                    rows.append(dict(scenario=sc, target=tgt["name"], seed=seed, method=m,
                                     acc=acc, macro_f1=f1))
                if (tgt["name"], seed) in old:
                    diffs.append(abs(res["M3L1"][0] - old[(tgt["name"], seed)]))
                print(f"  种子 {seed} {tgt['name']}  " + "  ".join(f"{m}={v[0]:.4f}" for m, v in res.items()))
        check[sc] = dict(
            m3l1_compared=len(diffs),
            m3l1_max_abs_diff_vs_main=max(diffs) if diffs else None,
            first_layer_sigA_over_sigH=dict(median=float(np.median(ratios)), min=float(np.min(ratios)),
                                            max=float(np.max(ratios))))
        print(f"  {sc}：M3L1 与已有主结果比较 {len(diffs)} 项，最大差 {check[sc]['m3l1_max_abs_diff_vs_main']}")
    for name, data in (("tta_m1l1.csv", rows), ("tta_m1l1_e0.csv", e0)):
        with open(RESULT_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"  -> {RESULT_DIR / name}")
    (RESULT_DIR / "tta_m1l1_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(f"  -> {RESULT_DIR / 'tta_m1l1_check.json'}")
    print(f"完成，用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()

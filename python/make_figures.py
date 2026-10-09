"""
生成论文图 3–7（png 300 dpi + pdf 矢量图），数据全部来自 results/ 下实验程序输出的文件
====================================================================================
  图 3  合成信号上的理论验证（命题 3：均值误差与 ε 成线性）               ← theory_check.json
  图 4  E1 主结果：各目标域上 M0–M4、B1–B3 的准确率（均值 ± 标准差）      ← tta_<场景>_summary.csv
  图 5  E2 校准数据污染比例 ε 与准确率（M1、M3、M3L1，开发目标域）         ← tta_<场景>_ablation.csv
  图 6  E3 每层窗口数 N 与准确率（M3、M3L1）                               ← tta_<场景>_ablation.csv
  图 7  E5 假设 A 检查：各层的尺度 R² 与均值 R²（每个目标域一条线）         ← tta_<场景>_e5_allseeds.json
  图 8  E4 校准前 k 层与准确率                                             ← tta_<场景>_ablation.csv
（图 1 方法框架、图 2 硬件平台照片由作者绘制/拍摄）
缺少哪个文件就跳过哪张图。

用法：python make_figures.py                  （默认场景 C1 C2 P）
输出：results/fig3_theory_synthetic.png/.pdf 等
"""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from config import RESULT_DIR  # noqa: E402

ORDER = ["M0", "M1", "M2", "M3", "M3L1", "M4", "B1", "B2", "B3", "B2*"]
LANG = "zh"      # --lang en 时改为英文标注
PAPER = False    # --paper 时不画总标题（论文中用图注）
LABEL_ZH = {"M0": "M0 无自适应", "M1": "M1 常规锚点", "M2": "M2 混合流(参考)", "M3": "M3 健康锚定(全部层)",
            "M3L1": "M3L1 本文(仅第一层)", "M4": "M4 有标签微调", "B1": "B1 DUA式", "B2": "B2 α-BN(α=0.5)",
            "B3": "B3 TENT", "B2*": "B2* α-BN(α=0.1，开发集选定)"}
LABEL_EN = {"M0": "M0 no adaptation", "M1": "M1 all-class anchor", "M2": "M2 mixed stream (ref.)",
            "M3": "M3 healthy anchor, all layers", "M3L1": "M3L1 proposed (first layer)",
            "M4": "M4 labelled fine-tuning", "B1": "B1 DUA-style", "B2": "B2 α-BN (α=0.5)", "B3": "B3 TENT",
            "B2*": "B2* α-BN (α=0.1, dev-selected)"}
LABEL = LABEL_ZH


def T(zh, en):
    return en if LANG == "en" else zh


def suptitle(fig, text, **kw):
    if not PAPER:
        fig.suptitle(text, **kw)


def set_font():
    """自动选择可用的中文字体（Windows：微软雅黑/黑体；Linux：Noto CJK）。"""
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "PingFang SC"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9


def read_csv(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    if LANG == "en":
        name += "_en"
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> results/{name}.png / .pdf")


def fig_main(scenarios):
    rows = []
    for sc in scenarios:
        rows += read_csv(RESULT_DIR / f"tta_{sc}_summary.csv") or []
    if not rows:
        print("跳过图 4：没有 summary 文件")
        return
    for sc in scenarios:                       # 补充实验中开发集选定 α 的 B2*（若有 _supp.csv）
        sup = read_csv(RESULT_DIR / f"tta_{sc}_supp.csv") or []
        for t in dict.fromkeys(r["target"] for r in sup if r["target"] != "E0"):
            v = np.array([100 * float(r["acc"]) for r in sup if r["target"] == t and r["method"] == "B2_a0.1"])
            if len(v):
                rows.append(dict(target=t, method="B2*", acc_mean=v.mean(), acc_std=v.std()))
    targets = list(dict.fromkeys(r["target"] for r in rows))
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(targets) + 2), 3.2))
    for j, m in enumerate(methods):
        mean = [next((float(r["acc_mean"]) for r in rows if r["target"] == t and r["method"] == m), np.nan)
                for t in targets]
        std = [next((float(r["acc_std"]) for r in rows if r["target"] == t and r["method"] == m), 0)
               for t in targets]
        ax.bar(np.arange(len(targets)) + (j - (len(methods) - 1) / 2) * w, mean, w, yerr=std, capsize=1.5,
               label=LABEL[m], hatch="//" if m == "M3L1" else None, edgecolor="black", linewidth=0.4)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel(T("准确率 (%)", "Accuracy (%)"))
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig4_main")


def _agg(rows, target, group):
    """把同一目标域、同一组的消融结果按 setting 汇总成 均值 ± 标准差（跨随机种子）。"""
    sel = [r for r in rows if r["target"] == target and r["group"] == group]
    keys = list(dict.fromkeys(r["setting"] for r in sel))
    xs, mu, sd, n = [], [], [], []
    for k in keys:
        v = np.array([100 * float(r["acc"]) for r in sel if r["setting"] == k])
        xs.append(float(k.split("=")[1]) if "=" in k else float(k.split("_")[1]))
        mu.append(v.mean())
        sd.append(v.std())
        n.append(len(v))
    return np.array(xs), np.array(mu), np.array(sd), (min(n) if n else 0)


def fig_ablation(scenarios):
    rows = []
    for sc in scenarios:
        rows += [dict(r, sc=sc) for r in (read_csv(RESULT_DIR / f"tta_{sc}_ablation.csv") or [])]
    if not rows:
        print("跳过图 5、图 6、图 8：没有 ablation 文件")
        return
    tg = list(dict.fromkeys(r["target"] for r in rows))
    if not any(r["group"] == "contamination_M3L1" for r in rows):
        print("提示：ablation 文件是旧版（没有 M3L1），请先运行 experiments_tta.py --scenario <场景> --ablation-only")
    style = {"M1": ("--s", "C1"), "M3": (":^", "C3"), "M3L1": ("-o", "C4")}

    # 图 5：污染比例 ε
    fig, axs = plt.subplots(1, len(tg), figsize=(2.6 * len(tg), 2.6), squeeze=False)
    nseeds = 0
    for a, t in zip(axs[0], tg):
        for m in ("M1", "M3", "M3L1"):
            x, mu, sd, n = _agg(rows, t, f"contamination_{m}")
            if len(x):
                nseeds = max(nseeds, n)
                a.errorbar(x, mu, sd, fmt=style[m][0], color=style[m][1], ms=3, capsize=2, lw=1,
                           label=LABEL[m].split(" ")[0])
        a.set_title(t, fontsize=8)
        a.set_xlabel(T("故障窗口比例 ε", "Fault-window fraction ε"))
        a.grid(alpha=0.3)
    axs[0][0].set_ylabel(T("准确率 (%)", "Accuracy (%)"))
    axs[0][0].legend(fontsize=7)
    suptitle(fig, T(f"校准数据被故障窗口污染（{nseeds} 个种子，均值 ± 标准差）", f"Contaminated calibration data ({nseeds} seeds, mean ± std)"), fontsize=8, y=1.02)
    save(fig, "fig5_contamination")

    # 图 6：每层窗口数 N
    fig, axs = plt.subplots(1, len(tg), figsize=(2.6 * len(tg), 2.6), squeeze=False)
    for a, t in zip(axs[0], tg):
        for grp, m in (("windows", "M3"), ("windows_M3L1", "M3L1")):
            x, mu, sd, n = _agg(rows, t, grp)
            if len(x):
                a.errorbar(x, mu, sd, fmt=style[m][0], color=style[m][1], ms=3, capsize=2, lw=1,
                           label=LABEL[m].split(" ")[0])
        a.set_xscale("log", base=2)
        a.set_title(t, fontsize=8)
        a.set_xlabel(T("每层校准窗口数 N", "Calibration windows per layer N"))
        a.grid(alpha=0.3)
    axs[0][0].set_ylabel(T("准确率 (%)", "Accuracy (%)"))
    axs[0][0].legend(fontsize=7)
    save(fig, "fig6_windows")

    # 图 8：校准前 k 层
    if any(r["group"] == "layers" for r in rows):
        fig, ax = plt.subplots(figsize=(4.2, 3))
        for t in tg:
            x, mu, sd, n = _agg(rows, t, "layers")
            if len(x):
                ax.errorbar(x, mu, sd, fmt="-o", ms=3, capsize=2, lw=1, label=t)
        ax.set_xlabel(T("校准的卷积层数 k（k=1 为 M3L1，k=全部 为 M3）", "Number of recalibrated conv layers k (k=1: M3L1; k=7: M3)"))
        ax.set_ylabel(T("准确率 (%)", "Accuracy (%)"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        save(fig, "fig8_layers")


def fig_e5(scenarios):
    """图 7：假设 A 在各层的成立程度（尺度 R²、第 2 层起的均值 R²），每个目标域一条线。
    优先用 _e5_allseeds.json（5 个种子，均值 ± 标准差），没有则用 _e5_summary.json（第一个种子）。"""
    data, src = {}, ""
    for sc in scenarios:
        for name in (f"tta_{sc}_e5_allseeds.json", f"tta_{sc}_e5_summary.json"):
            p = RESULT_DIR / name
            if p.exists():
                for d in json.loads(p.read_text(encoding="utf-8")):
                    if "r2_std_by_layer" in d:
                        data.setdefault(d["target"], []).append(d)
                src = name
                break
    if not data:
        print("跳过图 7：没有 e5 汇总文件")
        return
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for t, ds in data.items():
        L = min(len(d["r2_std_by_layer"]) for d in ds)
        sd_ = np.array([d["r2_std_by_layer"][:L] for d in ds])
        mn_ = np.array([d["r2_mean_by_layer"][:L] for d in ds])
        x = np.arange(1, L + 1)
        for a, v, sl in ((ax[0], sd_, slice(0, None)), (ax[1], mn_, slice(1, None))):
            med = np.clip(np.median(v, 0), -3, None)[sl]
            lo = np.clip(np.percentile(v, 25, 0), -3, None)[sl]
            hi = np.clip(np.percentile(v, 75, 0), -3, None)[sl]
            a.errorbar(x[sl], med, [med - lo, hi - med], fmt="-o", ms=3, capsize=2, lw=1, label=t)
    for a, ttl in ((ax[0], T("(a) 故障类标准差的预测 R²", "(a) R² of predicted fault-class std")),
                   (ax[1], T("(b) 故障类均值的预测 R²（第 2 层起）", "(b) R² of predicted fault-class mean (layer ≥ 2)"))):
        a.axhline(0, color="k", lw=0.6)
        a.set_ylim(-3.2, 1.1)
        a.set_xlabel(T("卷积层序号", "Conv layer index"))
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        a.set_title(ttl, fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel(T("R²（越接近 1 越符合假设 A）", "R² (1 = Assumption A holds)"))
    ax[0].legend(fontsize=6, ncol=2, loc="lower left")
    n = max(len(v) for v in data.values())
    suptitle(fig, T(f"假设 A 检查（{n} 个种子的中位数，误差线为四分位范围；低于 −3 的值画在下边界）", f"Assumption A check (median of {n} seeds, bars: interquartile range; values below −3 drawn at the bound)"), fontsize=8, y=1.02)
    print(f"  图 7 数据来源：{src} 等")
    save(fig, "fig7_assumption_a")


def fig_theory():
    p = RESULT_DIR / "theory_check.json"
    if not p.exists():
        print("跳过图 3：没有 theory_check.json（先运行 check_theory.py）")
        return
    s = json.loads(p.read_text(encoding="utf-8"))["summary"]
    eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].errorbar(eps, [v["mean"] for v in s["V3_err_by_eps"]], [v["std"] for v in s["V3_err_by_eps"]], fmt="-o",
                   ms=3, capsize=2)
    ax[0].set_xlabel("ε")
    ax[0].set_ylabel(T("均值误差 (σ 单位)", "Mean error (units of σ)"))
    ax[0].set_title(T("(a) 线性拟合", "(a) linear fit") + f" R²={s['V3_r2']['mean']:.3f}", fontsize=8)
    ax[1].errorbar(eps, [100 * v["mean"] for v in s["V3_acc_by_eps"]], [100 * v["std"] for v in s["V3_acc_by_eps"]],
                   fmt="-s", ms=3, capsize=2)
    ax[1].set_xlabel("ε")
    ax[1].set_ylabel(T("M3 准确率 (%)", "M3 accuracy (%)"))
    ax[1].set_title(T("(b) 合成数据", "(b) synthetic data"), fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    save(fig, "fig3_theory_synthetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["C1", "C2", "P"])
    ap.add_argument("--lang", default="zh", choices=["zh", "en"], help="图中文字的语言")
    ap.add_argument("--paper", action="store_true", help="论文用图：不画总标题")
    a = ap.parse_args()
    global LANG, PAPER, LABEL
    LANG, PAPER = a.lang, a.paper
    LABEL = LABEL_EN if LANG == "en" else LABEL_ZH
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    set_font()
    fig_main(a.scenarios)
    fig_ablation(a.scenarios)
    fig_e5(a.scenarios)
    fig_theory()


if __name__ == "__main__":
    main()

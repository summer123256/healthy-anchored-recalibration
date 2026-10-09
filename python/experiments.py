"""
一键运行论文中的全部实验（CWRU）
==============================
  E1 主对比实验：SVM / PlainCNN / WDCNN / 本文 Tiny，多随机种子，报告均值±标准差
  E2 抗噪实验：SNR = -4 … 10 dB
  E3 跨负载实验：负载 i 训练、负载 j 测试（12 组）
  E4 消融实验：首层卷积核长度、深度可分离卷积、噪声增强
  E5 混淆矩阵

用法：
  python experiments.py            # 完整版（CPU 约数小时，视电脑而定）
  python experiments.py --quick    # 快速版（少轮数、少种子，先确认流程跑通）
  python experiments.py --only E2 E3
结果：results/*.csv 与 results/*.png
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from baseline_svm import run_svm  # noqa: E402
from config import CKPT_DIR, CWRU_CLASSES, RESULT_DIR  # noqa: E402
from data_cwru import build_cwru  # noqa: E402
from models import build_model, count_macs, count_params  # noqa: E402
from train import evaluate, train_model  # noqa: E402

SNRS = [-4, -2, 0, 2, 4, 6, 8, 10]
NC = len(CWRU_CLASSES)


def sel(d, loads):
    m = np.isin(d[2], loads)
    return tuple(a[m] for a in d)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def fit(name, tr, va, ep, seed, noise_aug=False):
    m = build_model(name, NC)
    train_model(m, tr, va, epochs=ep, seed=seed, noise_aug=noise_aug, verbose=False)
    return m


def e1_main(ds, ep, seeds):
    print("E1 主对比实验")
    rows = []
    tr, va, te = ds["train"], ds["val"], ds["test"]
    for s in seeds:
        r, _ = run_svm(tr, te, seed=s)
        rows.append(dict(model="svm", seed=s, acc=r["acc"], macro_f1=r["macro_f1"], params="", macs=""))
        print(f"  svm seed{s}: {r['acc']:.4f}")
    for name in ["plaincnn", "wdcnn", "tiny"]:
        for s in seeds:
            m = fit(name, tr, va, ep, s)
            r = evaluate(m, te, NC)
            rows.append(dict(model=name, seed=s, acc=r["acc"], macro_f1=r["macro_f1"],
                             params=count_params(m), macs=count_macs(m)))
            print(f"  {name} seed{s}: {r['acc']:.4f}")
            if name == "tiny" and s == seeds[0]:
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=m.state_dict(), model="tiny", n_classes=NC,
                                classes=CWRU_CLASSES), CKPT_DIR / "cwru_tiny_main.pt")
                plot_cm(np.array(r["cm"]), CWRU_CLASSES, RESULT_DIR / "E5_confusion_tiny.png")
    write_csv(RESULT_DIR / "E1_main.csv", rows)
    summ = []
    for name in ["svm", "plaincnn", "wdcnn", "tiny"]:
        a = np.array([r["acc"] for r in rows if r["model"] == name])
        f = np.array([r["macro_f1"] for r in rows if r["model"] == name])
        p = next(r["params"] for r in rows if r["model"] == name)
        mc = next(r["macs"] for r in rows if r["model"] == name)
        summ.append(dict(model=name, acc_mean=a.mean(), acc_std=a.std(), f1_mean=f.mean(),
                         f1_std=f.std(), params=p, macs=mc))
    write_csv(RESULT_DIR / "E1_summary.csv", summ)


def e2_noise(ds, ep, seed):
    print("E2 抗噪实验")
    tr, va, te = ds["train"], ds["val"], ds["test"]
    rows = []
    cfgs = [("wdcnn", False), ("tiny", False), ("tiny", True)]
    for name, na in cfgs:
        m = fit(name, tr, va, ep, seed, noise_aug=na)
        for snr in SNRS + [None]:
            r = evaluate(m, te, NC, snr=snr, seed=seed)
            rows.append(dict(model=name + ("+noise_aug" if na else ""), snr="clean" if snr is None else snr,
                             acc=r["acc"], macro_f1=r["macro_f1"]))
        print(f"  {name}{'+NA' if na else ''} 完成")
    for snr in SNRS:
        r, _ = run_svm(tr, te, snr=snr, seed=seed)
        rows.append(dict(model="svm", snr=snr, acc=r["acc"], macro_f1=r["macro_f1"]))
    write_csv(RESULT_DIR / "E2_noise.csv", rows)
    plt.figure(figsize=(6, 4))
    for mname in dict.fromkeys(r["model"] for r in rows):
        pts = [(r["snr"], r["acc"]) for r in rows if r["model"] == mname and r["snr"] != "clean"]
        plt.plot([p[0] for p in pts], [p[1] * 100 for p in pts], marker="o", label=mname)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "E2_noise.png", dpi=300)
    plt.close()


def e3_crossload(ds, ep, seed):
    print("E3 跨负载实验")
    rows = []
    for name in ["wdcnn", "tiny"]:
        for i in range(4):
            tr, va = sel(ds["train"], [i]), sel(ds["val"], [i])
            m = fit(name, tr, va, ep, seed)
            for j in range(4):
                if i == j:
                    continue
                r = evaluate(m, sel(ds["test"], [j]), NC)
                rows.append(dict(model=name, train_load=i, test_load=j, acc=r["acc"], macro_f1=r["macro_f1"]))
            print(f"  {name} 负载{i} 完成")
    write_csv(RESULT_DIR / "E3_crossload.csv", rows)


def e4_ablation(ds, ep, seeds):
    print("E4 消融实验")
    tr, va, te = ds["train"], ds["val"], ds["test"]
    rows = []
    cfgs = [("tiny_k8", False), ("tiny_k16", False), ("tiny_k32", False), ("tiny", False),
            ("tiny_k128", False), ("tiny_std", False), ("tiny", True)]
    for name, na in cfgs:
        for s in seeds:
            m = fit(name, tr, va, ep, s, noise_aug=na)
            r = evaluate(m, te, NC)
            r0 = evaluate(m, te, NC, snr=0, seed=s)
            rows.append(dict(variant=name + ("+noise_aug" if na else ""), seed=s, acc=r["acc"],
                             acc_snr0=r0["acc"], params=count_params(m), macs=count_macs(m)))
        print(f"  {name}{'+NA' if na else ''} 完成")
    write_csv(RESULT_DIR / "E4_ablation.csv", rows)


def plot_cm(cm, names, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                     color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", default=["E1", "E2", "E3", "E4"])
    a = ap.parse_args()
    ep = 5 if a.quick else 60
    seeds = [0] if a.quick else [0, 1, 2, 3, 4]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ds = build_cwru()
    if "E1" in a.only:
        e1_main(ds, ep, seeds)
    if "E2" in a.only:
        e2_noise(ds, ep, seeds[0])
    if "E3" in a.only:
        e3_crossload(ds, ep, seeds[0])
    if "E4" in a.only:
        e4_ablation(ds, ep, seeds[:3])
    (RESULT_DIR / "run_info.json").write_text(json.dumps(dict(epochs=ep, seeds=seeds), indent=2))
    print("全部完成。")


if __name__ == "__main__":
    main()

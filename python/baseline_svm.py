"""
传统方法基线：手工特征 + SVM
==========================
特征（共 27 维）：
  时域 11 维：均值绝对值、均方根、标准差、峰值、峰峰值、偏度、峭度、
             峰值因子、波形因子、脉冲因子、裕度因子
  频域 16 维：幅值谱均分为 16 个频带的能量占比
输入与 CNN 完全相同（int8 预处理后的信号），保证对比公平。

用法：python baseline_svm.py [--train-loads 0 --test-loads 3] [--snr 0]
"""
import argparse
import json

import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RESULT_DIR
from preprocess import windows_to_input


def features(X):
    x = X.reshape(len(X), -1).astype(np.float64)
    ab = np.abs(x)
    mean_abs = ab.mean(1)
    rms = np.sqrt((x ** 2).mean(1))
    std = x.std(1)
    peak = ab.max(1)
    p2p = x.max(1) - x.min(1)
    sk = skew(x, axis=1)
    ku = kurtosis(x, axis=1)
    eps = 1e-12
    crest = peak / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = peak / (mean_abs + eps)
    margin = peak / (np.sqrt(ab).mean(1) ** 2 + eps)
    spec = np.abs(np.fft.rfft(x, axis=1))[:, 1:]
    bands = np.array_split(spec, 16, axis=1)
    be = np.stack([b.sum(1) for b in bands], 1)
    be = be / (be.sum(1, keepdims=True) + eps)
    return np.column_stack([mean_abs, rms, std, peak, p2p, sk, ku, crest, shape, impulse, margin, be])


def run_svm(train, test, snr=None, seed=0):
    Xtr, ytr, _ = train
    Xte, yte, _ = test
    ftr = features(windows_to_input(Xtr)[0])
    rng = np.random.default_rng(seed)
    fte = features(windows_to_input(Xte, snr_db=snr, rng=rng)[0])
    clf = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                       {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}, cv=3, n_jobs=-1)
    clf.fit(ftr, ytr)
    p = clf.predict(fte)
    return dict(acc=float((p == yte).mean()), macro_f1=float(f1_score(yte, p, average="macro")),
                best_params=clf.best_params_), clf


def main():
    from data_cwru import build_cwru
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--test-loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--snr", type=float, default=None)
    a = ap.parse_args()
    ds = build_cwru(tuple(sorted(set(a.train_loads) | set(a.test_loads))))
    sel = lambda d, ls: tuple(x[np.isin(d[2], ls)] for x in d)  # noqa: E731
    res, _ = run_svm(sel(ds["train"], a.train_loads), sel(ds["test"], a.test_loads), a.snr)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "svm_baseline.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

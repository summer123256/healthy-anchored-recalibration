"""
帕德博恩大学（PU）轴承数据读取与划分
=================================
数据：6 个轴承（正常 K001、K002；外圈 KA04、KA15；内圈 KI04、KI14），4 种工况，每种工况 20 次测量、每次 4 秒。
文件放在 data/pu/<轴承代号>/ 下（解压时多套一层同名文件夹也可以，这里会递归查找）。

划分原则：按“测量序号”划分，不同用途之间没有任何重叠的测量。
  build_pu_source()        源域工况 N15_M07_F10：第 1–12 次训练 / 13–16 验证 / 17–20 测试
  build_pu_target("P1")    目标域工况：第 1–6 次为校准段（M1/M3 只取其中正常类）/ 第 7–20 次测试
      P1 = N09_M07_F10（换转速），P2 = N15_M01_F10（换负载扭矩），P3 = N15_M07_F04（换径向力）

处理：vibration_1（64 kHz）→ 抗混叠 FIR 滤波并 4 倍降采样到 16 kHz → 乘以统一比例换算成 int16 计数
     → 按 WIN=1024 滑窗（训练段步长 TRAIN_STRIDE，其余 EVAL_STRIDE）。
比例：用源域训练数据 |x| 的 99.99% 分位数映射到 PU_INT16_TARGET，源域和目标域用同一个比例。

运行 python data_pu.py 检查文件是否齐全、打印样本数。
"""
from functools import lru_cache

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

from config import (EVAL_STRIDE, PU_BEARINGS, PU_CLASSES, PU_DECIMATE, PU_DIR, PU_INT16_TARGET,
                    PU_SOURCE, PU_SOURCE_SPLIT, PU_TARGET_SPLIT, PU_TARGETS, TRAIN_STRIDE, WIN)
from data_cwru import sliding_windows


def find_file(cond, bearing, idx):
    """在 data/pu/ 下递归查找 <工况>_<轴承>_<序号>.mat"""
    name = f"{cond}_{bearing}_{idx}.mat"
    hits = list(PU_DIR.rglob(name))
    if not hits:
        raise FileNotFoundError(f"找不到 {name}：请把 {bearing}.rar 解压到 {PU_DIR / bearing}")
    return hits[0]


def _vibration_from_mat(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise KeyError(f"{path.name} 里没有数据变量")
    obj = mat[path.stem] if path.stem in mat else mat[keys[0]]    # 个别文件变量名与文件名不一致
    names = []
    for ch in np.atleast_1d(getattr(obj, "Y")):
        name = str(getattr(ch, "Name", "")).strip()
        names.append(name)
        if name == "vibration_1":
            return np.asarray(getattr(ch, "Data"), dtype=np.float64).ravel()
    raise KeyError(f"{path.name} 中没有 vibration_1 通道，实际通道：{names}")


@lru_cache(maxsize=None)
def load_signal(cond, bearing, idx):
    """读取一次测量的振动信号，降采样到 16 kHz，单位与原始文件相同（未换算）。"""
    x = _vibration_from_mat(find_file(cond, bearing, idx))
    return decimate(x, PU_DECIMATE, ftype="fir", zero_phase=True).astype(np.float32)


@lru_cache(maxsize=1)
def int16_scale():
    """由源域训练数据确定统一换算比例（源域、目标域相同）。"""
    peaks = [np.percentile(np.abs(load_signal(PU_SOURCE, b, i)), 99.99)
             for bs in PU_BEARINGS.values() for b in bs for i in PU_SOURCE_SPLIT["train"]]
    return float(PU_INT16_TARGET / max(np.median(peaks), 1e-12))


def _build(cond, split, strides):
    scale = int16_scale()
    out = {}
    for part, idxs in split.items():
        X, y, g = [], [], []
        for label, cname in enumerate(PU_CLASSES):
            for bi, bearing in enumerate(PU_BEARINGS[cname]):
                for i in idxs:
                    w = sliding_windows(load_signal(cond, bearing, i) * scale, WIN, strides[part])
                    X.append(w)
                    y.append(np.full(len(w), label, dtype=np.int64))
                    g.append(np.full(len(w), 10 * label + bi, dtype=np.int64))   # 轴承编号（分析用）
        out[part] = (np.concatenate(X), np.concatenate(y), np.concatenate(g))
    return out


def build_pu_source():
    """源域：{'train'|'val'|'test': (X, y, 轴承编号)}，X 为浮点窗口（int16 计数单位）。"""
    return _build(PU_SOURCE, PU_SOURCE_SPLIT,
                  {"train": TRAIN_STRIDE, "val": EVAL_STRIDE, "test": EVAL_STRIDE})


def build_pu_target(name):
    """目标域 P1 / P2 / P3：{'calib'|'test': (X, y, 轴承编号)}"""
    return _build(PU_TARGETS[name], PU_TARGET_SPLIT, {"calib": EVAL_STRIDE, "test": EVAL_STRIDE})


def check_files():
    missing = []
    for cond in [PU_SOURCE] + list(PU_TARGETS.values()):
        for bs in PU_BEARINGS.values():
            for b in bs:
                for i in range(1, 21):
                    try:
                        find_file(cond, b, i)
                    except FileNotFoundError as e:
                        missing.append(str(e))
    return missing


if __name__ == "__main__":
    miss = check_files()
    if miss:
        print(f"缺少 {len(miss)} 个文件，例如：")
        for m in miss[:5]:
            print("  " + m)
    else:
        print("文件齐全：6 个轴承 × 4 种工况 × 20 次测量")
        print(f"换算比例：{int16_scale():.4g}（原始单位 → int16 计数）")
        for name, ds in [("源域 " + PU_SOURCE, build_pu_source())] + \
                        [(f"{k} {v}", build_pu_target(k)) for k, v in PU_TARGETS.items()]:
            print(name)
            for s, (X, y, _) in ds.items():
                print(f"  {s:5s}: {len(y):6d} 个窗口，每类 {np.bincount(y, minlength=3).tolist()}")

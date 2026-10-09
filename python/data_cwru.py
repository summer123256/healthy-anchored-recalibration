"""
CWRU 数据读取与划分
==================
关键原则：先按时间把每条原始信号切段，再在各段内部滑窗。
训练、验证、测试（以及目标域的“校准段”）之间没有任何重叠采样点，避免数据泄露。

两种用法：
  build_cwru(loads, channel)          源域：训练 60% / 验证 20% / 测试 20%
  build_cwru_target(loads, channel)   目标域：校准段 30%（只用于重校准）/ 测试段 70%（C2，负载与源域不同）
  build_cwru_target(loads, "FE", synchronous=True)
                                      C1：与源域同步记录，校准段 = 60–80%，测试段 = 80–100%（不与源域训练段重叠）
channel："DE"（驱动端加速度计）或 "FE"（风扇端加速度计）
"""
import re
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

from config import (CWRU_CLASSES, CWRU_DIR, CWRU_FILES, CWRU_INT16_SCALE,
                    CWRU_NORMAL_DECIMATE, CWRU_NORMAL_DECIMATE_FACTOR,
                    EVAL_STRIDE, SPLIT_RATIO, TRAIN_STRIDE, WIN)

SOURCE_PARTS = (("train", 0.0, SPLIT_RATIO[0], TRAIN_STRIDE),
                ("val", SPLIT_RATIO[0], SPLIT_RATIO[0] + SPLIT_RATIO[1], EVAL_STRIDE),
                ("test", SPLIT_RATIO[0] + SPLIT_RATIO[1], 1.0, EVAL_STRIDE))
TARGET_PARTS = (("calib", 0.0, 0.3, EVAL_STRIDE),
                ("test", 0.3, 1.0, EVAL_STRIDE))
# C1（DE→FE）的源域与目标域来自同一组同步记录。为避免目标域测试段与源域训练段在时间上重叠，
# 目标域只取源域验证段、测试段对应的时间：校准段 = 源域验证段时间，测试段 = 源域测试段时间。
SYNC_TARGET_PARTS = (("calib", SPLIT_RATIO[0], SPLIT_RATIO[0] + SPLIT_RATIO[1], EVAL_STRIDE),
                     ("test", SPLIT_RATIO[0] + SPLIT_RATIO[1], 1.0, EVAL_STRIDE))

_cache = {}


def load_signal(num, channel="DE"):
    """读取一个 .mat 文件中指定加速度计的信号，单位 g。"""
    key_cache = (num, channel)
    if key_cache in _cache:
        return _cache[key_cache]
    path = CWRU_DIR / f"{num}.mat"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 download_cwru.py")
    mat = loadmat(path)
    keys = [k for k in mat.keys() if re.fullmatch(rf"X\d+_{channel}_time", k)]
    if not keys:
        raise KeyError(f"{path.name} 中没有 *_{channel}_time 变量，实际变量：{list(mat.keys())}")
    want = f"X{num:03d}_{channel}_time"
    key = want if want in keys else keys[0]
    if key != want:
        warnings.warn(f"{path.name}: 未找到 {want}，改用 {key}（CWRU 个别文件命名不规范）")
    x = mat[key].astype(np.float64).ravel()
    _cache[key_cache] = x
    return x


def load_de_signal(num):
    """兼容旧代码。"""
    return load_signal(num, "DE")


def sliding_windows(x, win, stride):
    if len(x) < win:
        return np.empty((0, win), dtype=np.float32)
    idx = np.arange(0, len(x) - win + 1, stride)
    return np.stack([x[i:i + win] for i in idx]).astype(np.float32)


def split_signal(x, ratio=SPLIT_RATIO):
    n = len(x)
    a = int(n * ratio[0])
    b = int(n * (ratio[0] + ratio[1]))
    return x[:a], x[a:b], x[b:]


def _build(loads, channel, parts):
    out = {p[0]: ([], [], []) for p in parts}
    for label, cname in enumerate(CWRU_CLASSES):
        for load in loads:
            x = load_signal(CWRU_FILES[cname][load], channel)
            if cname == "Normal" and CWRU_NORMAL_DECIMATE:
                x = decimate(x, CWRU_NORMAL_DECIMATE_FACTOR, ftype="fir", zero_phase=True)
            x = x * CWRU_INT16_SCALE
            n = len(x)
            for name, a, b, stride in parts:
                w = sliding_windows(x[int(n * a):int(n * b)], WIN, stride)
                out[name][0].append(w)
                out[name][1].append(np.full(len(w), label, dtype=np.int64))
                out[name][2].append(np.full(len(w), load, dtype=np.int64))
    return {s: (np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(v[2]))
            for s, v in out.items()}


def build_cwru(loads=(0, 1, 2, 3), channel="DE"):
    """源域：{'train'|'val'|'test': (X, y, load)}。X 为浮点窗口，单位“int16 计数”。"""
    return _build(tuple(loads), channel, SOURCE_PARTS)


def build_cwru_target(loads=(0, 1, 2, 3), channel="FE", synchronous=False):
    """目标域：{'calib'|'test': (X, y, load)}。校准段只用于重校准（M1/M3 只取其中的正常类）。
    synchronous=True：目标域与源域是同一组记录的另一通道（C1），按 SYNC_TARGET_PARTS 划分。"""
    return _build(tuple(loads), channel, SYNC_TARGET_PARTS if synchronous else TARGET_PARTS)


def check_channels(channel="FE"):
    """检查 40 个文件是否都含指定通道。"""
    missing = []
    for cname in CWRU_CLASSES:
        for load, num in CWRU_FILES[cname].items():
            try:
                load_signal(num, channel)
            except (KeyError, FileNotFoundError) as e:
                missing.append(f"{num}.mat：{e}")
    return missing


def summary(ds):
    for s, (X, y, _) in ds.items():
        counts = np.bincount(y, minlength=len(CWRU_CLASSES))
        print(f"{s:5s}: {len(y):6d} 个样本，每类 {counts.tolist()}")


if __name__ == "__main__":
    print("源域（驱动端 DE）：")
    summary(build_cwru())
    miss = check_channels("FE")
    print("风扇端 FE 通道检查：" + ("全部存在" if not miss else f"缺失 {len(miss)} 个"))
    for m in miss:
        print("  " + m)
    # 论文表 2 需要的目标域窗口数
    if not miss:
        print("目标域 C1（风扇端 FE，0–3 HP，同步记录划分）：")
        summary(build_cwru_target((0, 1, 2, 3), "FE", synchronous=True))
    for h in (1, 2, 3):
        print(f"目标域 C2-{h}HP（驱动端 DE）：")
        summary(build_cwru_target((h,), "DE"))
    print("（场景 C2 的源域只用 0 HP，训练时由 experiments_tta.py 自动选取）")

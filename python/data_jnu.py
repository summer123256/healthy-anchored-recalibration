"""
江南大学（JNU）轴承数据读取与划分（补充验证：换转速）
================================================
数据：转速 600 / 800 / 1000 r/min，每个转速 4 类（正常、内圈、外圈、滚动体），每类一个连续记录，采样率 50 kHz。
文件放在 data/jnu/ 下（git clone 时多一层文件夹也可以，这里会递归查找）。

划分原则：先在原始信号上按时间划分，再对每一段分别滤波降采样，各段之间不共享任何原始采样点。
  build_jnu_source()        源域 1000 r/min：前 60% 训练 / 中间 20% 验证 / 最后 20% 测试
  build_jnu_target("J1")    目标域：前 30% 为校准段（只取其中正常类）/ 其余 70% 测试
      J1 = 800 r/min，J2 = 600 r/min；两者都没有参与任何设计选择，全部作为留出目标域

处理：每类取前 JNU_SAMPLES 点（各类等长）→ 按比例切段 → 每段抗混叠 FIR 滤波并 3 倍降采样
     → 乘以统一比例换算成 int16 计数 → 按 WIN=1024 滑窗（训练段步长 TRAIN_STRIDE，其余 EVAL_STRIDE）。
比例：用源域训练段 |x| 的 99.99% 分位数映射到 JNU_INT16_TARGET，源域和目标域用同一个比例。

运行 python data_jnu.py 检查文件是否齐全、打印样本数。
"""
from functools import lru_cache

import numpy as np
from scipy.signal import decimate

from config import (EVAL_STRIDE, JNU_CLASSES, JNU_DECIMATE, JNU_DIR, JNU_INT16_TARGET, JNU_PREFIX,
                    JNU_SAMPLES, JNU_SOURCE, JNU_SOURCE_SPLIT, JNU_TARGET_CALIB, JNU_TARGETS,
                    TRAIN_STRIDE, WIN)
from data_cwru import sliding_windows


def find_file(cls, rpm):
    name = JNU_PREFIX[cls].format(rpm=rpm) + ".csv"
    hits = list(JNU_DIR.rglob(name))
    if not hits:
        raise FileNotFoundError(f"找不到 {name}：请把 JNU 数据集的 12 个 .csv 放到 {JNU_DIR}")
    return hits[0]


@lru_cache(maxsize=None)
def load_raw(cls, rpm):
    """读取一个记录的前 JNU_SAMPLES 点（原始单位，50 kHz）。"""
    with open(find_file(cls, rpm), encoding="utf-8") as f:
        x = np.array([float(v) for v in f.read().split()], dtype=np.float64)
    if len(x) < JNU_SAMPLES:
        raise ValueError(f"{find_file(cls, rpm).name} 只有 {len(x)} 点，少于 {JNU_SAMPLES}")
    return x[:JNU_SAMPLES]


def _segments(x, ratios):
    """按比例在原始信号上切段，再分别降采样（各段互不重叠，也不共享滤波支撑区）。"""
    n = len(x)
    edges = np.round(np.cumsum([0] + list(ratios)) * n).astype(int)
    return [decimate(x[a:b], JNU_DECIMATE, ftype="fir", zero_phase=True).astype(np.float32)
            for a, b in zip(edges[:-1], edges[1:])]


@lru_cache(maxsize=1)
def int16_scale():
    peaks = [np.percentile(np.abs(_segments(load_raw(c, JNU_SOURCE), JNU_SOURCE_SPLIT)[0]), 99.99)
             for c in JNU_CLASSES]
    return float(JNU_INT16_TARGET / max(np.median(peaks), 1e-12))


def _build(rpm, ratios, parts, strides):
    scale = int16_scale()
    out = {p: ([], [], []) for p in parts}
    for label, cls in enumerate(JNU_CLASSES):
        segs = _segments(load_raw(cls, rpm), ratios)
        for p, seg in zip(parts, segs):
            w = sliding_windows(seg * scale, WIN, strides[p])
            out[p][0].append(w)
            out[p][1].append(np.full(len(w), label, dtype=np.int64))
            out[p][2].append(np.full(len(w), label, dtype=np.int64))
    return {p: tuple(np.concatenate(v) for v in out[p]) for p in parts}


def build_jnu_source():
    """源域：{'train'|'val'|'test': (X, y, 类别)}，X 为浮点窗口（int16 计数单位）。"""
    return _build(JNU_SOURCE, JNU_SOURCE_SPLIT, ("train", "val", "test"),
                  {"train": TRAIN_STRIDE, "val": EVAL_STRIDE, "test": EVAL_STRIDE})


def build_jnu_target(name):
    """目标域 J1 / J2：{'calib'|'test': (X, y, 类别)}"""
    return _build(JNU_TARGETS[name], (JNU_TARGET_CALIB, 1 - JNU_TARGET_CALIB), ("calib", "test"),
                  {"calib": EVAL_STRIDE, "test": EVAL_STRIDE})


def check_files():
    missing = []
    for rpm in [JNU_SOURCE] + list(JNU_TARGETS.values()):
        for c in JNU_CLASSES:
            try:
                find_file(c, rpm)
            except FileNotFoundError as e:
                missing.append(str(e))
    return missing


if __name__ == "__main__":
    miss = check_files()
    if miss:
        print(f"缺少 {len(miss)} 个文件，例如：{miss[0]}")
    else:
        print("文件齐全：3 种转速 × 4 类")
        print(f"换算比例：{int16_scale():.4g}（原始单位 → int16 计数）")
        for name, ds in [(f"源域 {JNU_SOURCE} r/min", build_jnu_source())] + \
                        [(f"{k} {v} r/min", build_jnu_target(k)) for k, v in JNU_TARGETS.items()]:
            print(name)
            for s, (X, y, _) in ds.items():
                print(f"  {s}: {len(y)} 个窗口，各类 {np.bincount(y, minlength=len(JNU_CLASSES)).tolist()}")

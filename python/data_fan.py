"""
自建风扇数据集读取与划分
======================
fan_collect.py 采集的文件：data/fan/<域>_<状态>_<用途>_<日期时间>.npz
  域   ：S0（源域）、F1 重新安装、F2 换设备、F3 换位置、F4 换转速
  状态 ：Normal / Imbalance / Looseness / BladeDamage
  用途 ：s1、s2（源域两个会话）、test（目标域测试）、calib（目标域校准，只录正常状态）
  data : int16 数组 (N, 3)，ADXL345 的 X/Y/Z 原始值（3.9 mg/LSB，±16 g 全分辨率）

build_fan_source()        源域 S0：每段录音按时间 60/20/20 切成训练/验证/测试
build_fan_target("F2")    目标域：
    calib_healthy  单独录制的正常状态数据（M1、M3 用）
    pool           各状态测试录音的前 30%（M2 混合数据流、M4 有标签微调用）
    test           各状态测试录音的后 70%（只用于测试）
"""
import re

import numpy as np

from config import EVAL_STRIDE, FAN_AXIS, FAN_CLASSES, FAN_DIR, SPLIT_RATIO, TRAIN_STRIDE, WIN
from data_cwru import sliding_windows

DOMAINS = ["S0", "F1", "F2", "F3", "F4"]
_NAME = re.compile(r"(?P<dom>S0|F[1-4])_(?P<cls>[A-Za-z]+)_(?P<use>s1|s2|test|calib)_.*\.npz")


def list_recordings(domain=None, use=None):
    recs = []
    for p in sorted(FAN_DIR.glob("*.npz")):
        m = _NAME.fullmatch(p.name)
        if not m or m["cls"] not in FAN_CLASSES:
            continue
        if domain and m["dom"] != domain:
            continue
        if use and m["use"] not in (use if isinstance(use, (list, tuple)) else [use]):
            continue
        recs.append((p, FAN_CLASSES.index(m["cls"]), m["dom"], m["use"]))
    return recs


def _load(path, axis):
    return np.load(path)["data"][:, axis].astype(np.float64)


def _pack(parts):
    res = {}
    for k, v in parts.items():
        if not v[0]:
            raise ValueError(f"'{k}' 为空：请检查 data/fan 下是否已采集对应文件")
        res[k] = (np.concatenate(v[0]), np.concatenate(v[1]), np.concatenate(v[2]))
    return res


def build_fan_source(axis=FAN_AXIS):
    recs = list_recordings("S0", ["s1", "s2"])
    if not recs:
        raise FileNotFoundError(f"{FAN_DIR} 中没有 S0 源域文件，请先运行 fan_collect.py")
    out = {s: ([], [], []) for s in ("train", "val", "test")}
    a, b = SPLIT_RATIO[0], SPLIT_RATIO[0] + SPLIT_RATIO[1]
    for path, label, _, use in recs:
        x = _load(path, axis)
        n = len(x)
        for name, lo, hi, stride in (("train", 0, a, TRAIN_STRIDE), ("val", a, b, EVAL_STRIDE),
                                     ("test", b, 1, EVAL_STRIDE)):
            w = sliding_windows(x[int(n * lo):int(n * hi)], WIN, stride)
            out[name][0].append(w)
            out[name][1].append(np.full(len(w), label, dtype=np.int64))
            out[name][2].append(np.full(len(w), 1 if use == "s1" else 2, dtype=np.int64))
    return _pack(out)


def build_fan_target(domain, axis=FAN_AXIS, pool_frac=0.3):
    recs = list_recordings(domain, ["test", "calib"])
    if not recs:
        raise FileNotFoundError(f"没有 {domain} 的采集文件")
    out = {s: ([], [], []) for s in ("calib_healthy", "pool", "test")}
    for path, label, _, use in recs:
        x = _load(path, axis)
        if use == "calib":
            if label != 0:
                raise ValueError(f"{path.name}：校准数据必须是正常状态（Normal）")
            w = sliding_windows(x, WIN, EVAL_STRIDE)
            out["calib_healthy"][0].append(w)
            out["calib_healthy"][1].append(np.zeros(len(w), dtype=np.int64))
            out["calib_healthy"][2].append(np.zeros(len(w), dtype=np.int64))
            continue
        cut = int(len(x) * pool_frac)
        for name, seg in (("pool", x[:cut]), ("test", x[cut:])):
            w = sliding_windows(seg, WIN, EVAL_STRIDE)
            out[name][0].append(w)
            out[name][1].append(np.full(len(w), label, dtype=np.int64))
            out[name][2].append(np.zeros(len(w), dtype=np.int64))
    return _pack(out)


if __name__ == "__main__":
    for dom in DOMAINS:
        try:
            ds = build_fan_source() if dom == "S0" else build_fan_target(dom)
        except (FileNotFoundError, ValueError) as e:
            print(f"{dom}: {e}")
            continue
        for s, (X, y, _) in ds.items():
            print(f"{dom} {s:14s} {len(y):5d} 个窗口，每类 {np.bincount(y, minlength=len(FAN_CLASSES)).tolist()}")

"""
预处理（与 STM32 端 tc_preprocess() 逐位一致）
=============================================
流程：int16 原始信号 -> 去均值 -> 最大绝对值归一化 -> int8 [-127, 127]

所有运算只用整数，舍入规则为“四舍五入、远离零”，C 端完全相同，
这样电脑上的结果和单片机上的结果可以做到逐位一致（bit-exact）。
"""
import numpy as np

from config import WIN


def div_round(a, b):
    """整数除法，四舍五入（远离零）。与 C 端 tc_div_round() 完全一致。
    要求 b > 0。a、b 可为 numpy 整数数组或 Python int。"""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    half = b // 2
    pos = (a + half) // b
    neg = -((-a + half) // b)
    return np.where(a >= 0, pos, neg)


def to_int16(x):
    """浮点（已乘好比例）-> int16，四舍五入并饱和。"""
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def preprocess_int16(x16):
    """x16: int16 数组，形状 (..., WIN)。返回 int8 数组，形状相同。"""
    x = np.asarray(x16, dtype=np.int64)
    assert x.shape[-1] == WIN, f"窗口长度必须为 {WIN}"
    mean = div_round(x.sum(axis=-1), WIN)
    d = x - mean[..., None]
    m = np.abs(d).max(axis=-1)
    m_safe = np.where(m == 0, 1, m)
    q = div_round(d * 127, m_safe[..., None])
    q = np.where((m == 0)[..., None], 0, q)
    return np.clip(q, -127, 127).astype(np.int8)


def int8_to_model_input(q):
    """int8 -> 神经网络浮点输入（除以 127，范围 [-1, 1]），并加通道维。"""
    return (np.asarray(q, dtype=np.float32) / 127.0)[..., None, :]


def add_awgn(x, snr_db, rng):
    """按信噪比给每个窗口加高斯白噪声。x: (N, WIN) 浮点。"""
    p_sig = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    return x + rng.standard_normal(x.shape) * np.sqrt(p_noise)


def windows_to_input(xw, snr_db=None, rng=None):
    """完整输入管线：浮点窗口（int16 计数单位） -> [可选加噪] -> int16 -> int8 -> 浮点模型输入"""
    x = xw
    if snr_db is not None:
        x = add_awgn(x, snr_db, rng if rng is not None else np.random.default_rng(0))
    x16 = to_int16(x)
    return int8_to_model_input(preprocess_int16(x16)), x16

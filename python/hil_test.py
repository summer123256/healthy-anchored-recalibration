"""
硬件在环（HIL）测试
==================
电脑把测试集样本逐条通过串口发给 STM32，单片机完成 预处理 + int8 推理，
返回预测类别、各阶段时钟周期数和 logits。脚本统计：
  * 单片机上的测试准确率
  * 单片机输出与 Python 整数参考实现是否逐位一致
  * 预处理 / 推理耗时（毫秒）的均值、标准差、最大值

用法：
  python hil_test.py --port COM5 --model-dir export/tiny --n 500
  （Linux/macOS 下端口类似 /dev/ttyUSB0 或 /dev/tty.usbserial-xxx）
"""
import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from config import MCU_CLOCK_HZ, SERIAL_BAUD
from preprocess import preprocess_int16
from quant import int_forward
from serial_proto import (serial_open, CMD_INFER, CMD_INFO, CMD_SELFTEST, RSP_INFER, RSP_INFO,
                          RSP_SELFTEST, make_frame, parse_result, read_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--model-dir", default="export/tiny")
    ap.add_argument("--n", type=int, default=0, help="测试样本数，0 表示全部")
    a = ap.parse_args()

    d = Path(a.model_dir)
    with open(d / "qmodel.pkl", "rb") as f:
        obj = pickle.load(f)
    q = obj["qlayers"] if isinstance(obj, dict) else obj
    ts = np.load(d / "hil_testset.npz")
    x16, y = ts["x16"], ts["y"]
    n_cls = q[-1]["out_shape"][0]
    if a.n:
        idx = np.random.default_rng(0).choice(len(y), size=min(a.n, len(y)), replace=False)
        x16, y = x16[idx], y[idx]
    ref = int_forward(q, preprocess_int16(x16)[:, None, :])

    ser = serial_open(a.port, a.baud, timeout=3)
    time.sleep(0.3)
    ser.reset_input_buffer()

    ser.write(make_frame(CMD_INFO))
    cmd, pl = read_frame(ser)
    if cmd == RSP_INFO:
        print("单片机信息:", pl.decode(errors="replace"))
    ser.write(make_frame(CMD_SELFTEST))
    cmd, pl = read_frame(ser)
    if cmd == RSP_SELFTEST:
        print(f"片上自检：{pl[0]}/{pl[1]} 通过")

    rows, t0 = [], time.time()
    for i in range(len(y)):
        ser.write(make_frame(CMD_INFER, x16[i].astype("<i2").tobytes()))
        cmd, pl = read_frame(ser)
        if cmd != RSP_INFER:
            raise RuntimeError(f"意外的响应 0x{cmd:02X}")
        pred, cp, ci, logits = parse_result(pl, n_cls)
        rows.append(dict(idx=i, label=int(y[i]), pred=pred, cyc_pre=cp, cyc_inf=ci,
                         bit_exact=int(list(ref[i]) == logits)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(y)}")
    ser.close()

    pred = np.array([r["pred"] for r in rows])
    ci = np.array([r["cyc_inf"] for r in rows]) / MCU_CLOCK_HZ * 1e3
    cp = np.array([r["cyc_pre"] for r in rows]) / MCU_CLOCK_HZ * 1e3
    rep = dict(n=len(rows), acc_mcu=float((pred == y).mean()),
               bit_exact_rate=float(np.mean([r["bit_exact"] for r in rows])),
               infer_ms_mean=float(ci.mean()), infer_ms_std=float(ci.std()), infer_ms_max=float(ci.max()),
               pre_ms_mean=float(cp.mean()), total_ms_mean=float((ci + cp).mean()),
               wall_seconds=round(time.time() - t0, 1))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    (d / "hil_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    with open(d / "hil_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

"""
采集自建风扇数据集
================
单片机以 3200 Hz 读取 ADXL345 三轴数据并连续发送，电脑保存为 .npz。

文件名：<域>_<状态>_<用途>_<日期时间>.npz，保存在 data/fan/
  --domain  S0（源域）、F1 重新安装、F2 换设备、F3 换位置、F4 换转速
  --use     s1、s2（源域两个会话，各 120 秒）、test（目标域测试，60 秒）、calib（目标域校准，只录正常 60 秒）

示例：
  python fan_collect.py --port COM5 --domain S0 --cls Normal    --use s1 --seconds 120
  python fan_collect.py --port COM5 --domain S0 --cls Imbalance --use s1 --seconds 120
  python fan_collect.py --port COM5 --domain F2 --cls Normal    --use test  --seconds 60
  python fan_collect.py --port COM5 --domain F2 --cls Normal    --use calib --seconds 60
完整采集顺序见《实验操作手册》第六节阶段四（扇叶破损不可逆，最后采）。
"""
import argparse
import struct
import time
from datetime import datetime

import numpy as np

from config import FAN_CLASSES, FAN_DIR, FAN_FS, SERIAL_BAUD
from serial_proto import serial_open, CMD_STOP, CMD_STREAM, RSP_STREAM, make_frame, read_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--domain", required=True, choices=["S0", "F1", "F2", "F3", "F4"])
    ap.add_argument("--cls", required=True, choices=FAN_CLASSES)
    ap.add_argument("--use", required=True, choices=["s1", "s2", "test", "calib"])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--warmup", type=float, default=3, help="开头丢弃的秒数（等风扇转速稳定）")
    a = ap.parse_args()
    if a.use == "calib" and a.cls != "Normal":
        ap.error("校准数据（--use calib）只能采集正常状态（--cls Normal）")
    if (a.domain == "S0") != (a.use in ("s1", "s2")):
        ap.error("源域 S0 用 --use s1/s2；目标域 F1–F4 用 --use test/calib")

    ser = serial_open(a.port, a.baud, timeout=2)
    time.sleep(0.3)
    ser.write(make_frame(CMD_STOP))
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(make_frame(CMD_STREAM))

    need = int((a.seconds + a.warmup) * FAN_FS)
    chunks, got, last_seq, lost, overruns = [], 0, None, 0, 0
    t0 = time.time()
    try:
        while got < need:
            cmd, pl = read_frame(ser)
            if cmd != RSP_STREAM:
                continue
            seq, n, ovr = struct.unpack_from("<HBB", pl, 0)
            data = np.frombuffer(pl, dtype="<i2", count=3 * n, offset=4).reshape(n, 3)
            if last_seq is not None and seq != ((last_seq + 1) & 0xFFFF):
                lost += 1
            last_seq = seq
            overruns += ovr
            chunks.append(data.copy())
            got += n
            if len(chunks) % 200 == 0:
                print(f"  {got / FAN_FS:6.1f} s")
    finally:
        ser.write(make_frame(CMD_STOP))
        ser.close()

    data = np.concatenate(chunks)[int(a.warmup * FAN_FS):]
    fs_est = got / (time.time() - t0)
    FAN_DIR.mkdir(parents=True, exist_ok=True)
    path = FAN_DIR / f"{a.domain}_{a.cls}_{a.use}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez_compressed(path, data=data.astype(np.int16), label=a.cls, domain=a.domain, use=a.use,
                        fs=FAN_FS)
    print(f"已保存 {path}，{len(data)} 点，约 {len(data) / FAN_FS:.1f} s")
    print(f"实测平均采样率约 {fs_est:.0f} Hz（含串口延迟，仅供参考）")
    if lost or overruns:
        print(f"警告：丢帧 {lost} 次，FIFO 溢出 {overruns} 次。请检查波特率或减少其他串口输出，"
              f"必要时重新采集。")


if __name__ == "__main__":
    main()

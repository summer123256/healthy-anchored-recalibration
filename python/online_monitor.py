"""
在线诊断监视
===========
单片机自己采集 1024 点 -> 预处理 -> 推理，循环发送结果；电脑只负责显示和记录。
这是论文“实际应用验证”部分的数据来源（此时电脑不参与任何计算）。

用法：python online_monitor.py --port COM5 --classes fan --true-label Imbalance --seconds 60
按 Ctrl+C 结束。结果追加写入 results/online_log.csv
"""
import argparse
import csv
import time

from config import CWRU_CLASSES, FAN_CLASSES, MCU_CLOCK_HZ, PU_CLASSES, RESULT_DIR, SERIAL_BAUD
from serial_proto import (serial_open, CMD_ONLINE, CMD_STOP, RSP_ONLINE, make_frame, parse_result,
                          read_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--classes", default="fan", choices=["fan", "cwru", "pu"])
    ap.add_argument("--true-label", default="", help="当前风扇的真实状态，用于统计准确率")
    ap.add_argument("--seconds", type=float, default=0, help="0 表示一直运行")
    a = ap.parse_args()
    names = {"fan": FAN_CLASSES, "cwru": CWRU_CLASSES, "pu": PU_CLASSES}[a.classes]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log = RESULT_DIR / "online_log.csv"
    new = not log.exists()
    f = open(log, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "true", "pred", "infer_ms", "pre_ms"])

    ser = serial_open(a.port, a.baud, timeout=3)
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(make_frame(CMD_ONLINE))
    n = ok = 0
    t0 = time.time()
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            cmd, pl = read_frame(ser)
            if cmd != RSP_ONLINE:
                continue
            pred, cp, ci, _ = parse_result(pl, len(names))
            n += 1
            ok += int(names[pred] == a.true_label)
            w.writerow([f"{time.time():.3f}", a.true_label, names[pred],
                        f"{ci / MCU_CLOCK_HZ * 1e3:.2f}", f"{cp / MCU_CLOCK_HZ * 1e3:.2f}"])
            acc = f"  累计准确率 {ok / n:.3f}" if a.true_label else ""
            print(f"#{n:5d} 诊断结果：{names[pred]:12s} 推理 {ci / MCU_CLOCK_HZ * 1e3:6.2f} ms{acc}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.write(make_frame(CMD_STOP))
        ser.close()
        f.close()


if __name__ == "__main__":
    main()

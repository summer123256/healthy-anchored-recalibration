"""
单片机自主片上重校准（风扇台实测）
================================
风扇正常运转时运行本脚本：STM32 自己通过 ADXL345 逐层采集 N 个窗口，在芯片上完成校准，
返回使用的窗口数、总耗时和计算耗时。之后可用 online_monitor.py 做在线诊断。

用法：
  python device_calibrate.py --port COM5                       # 默认：每层 32 个窗口、健康锚点、校正尺度、全部层
  python device_calibrate.py --port COM5 --anchor all          # 常规做法（M1），用于对比
  python device_calibrate.py --port COM5 --reset               # 恢复出厂参数（清除校准）
  python device_calibrate.py --port COM5 --dump params.json    # 校准后导出各层参数
"""
import argparse
import json
import struct
import time

from config import MCU_CLOCK_HZ, SERIAL_BAUD
from serial_proto import (CMD_CALIBRATE, CMD_DUMP, CMD_INFO, CMD_RESET_PARAMS, RSP_CALIBRATE,
                          RSP_DUMP, RSP_INFO, RSP_RESET_PARAMS, expect, make_frame, parse_dump,
                          serial_open)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--n-per-layer", type=int, default=32)
    ap.add_argument("--anchor", default="healthy", choices=["healthy", "all"])
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--layers", type=int, default=1, help="只校准前几个卷积层：1 = M3L1（默认），0 = 全部层")
    ap.add_argument("--reset", action="store_true", help="只恢复出厂参数")
    ap.add_argument("--dump", default="", help="校准后把各层参数保存到该 json 文件")
    a = ap.parse_args()
    if not 1 <= a.n_per_layer <= 255:
        ap.error("--n-per-layer 取 1–255")

    est = a.n_per_layer * (a.layers or 7) * 1024 / 3200
    ser = serial_open(a.port, a.baud, timeout=est + 30)
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(make_frame(CMD_INFO))
    info = expect(ser, RSP_INFO).decode(errors="replace")
    print("单片机信息:", info)
    if a.reset:
        ser.write(make_frame(CMD_RESET_PARAMS))
        expect(ser, RSP_RESET_PARAMS)
        print("已恢复出厂参数")
        ser.close()
        return

    print(f"开始片上校准：请保持设备正常运转，预计约 {est:.0f} 秒…")
    ser.write(make_frame(CMD_CALIBRATE, bytes([a.n_per_layer, 0 if a.anchor == "healthy" else 1,
                                               0 if a.no_scale else 1, a.layers])))
    pl = expect(ser, RSP_CALIBRATE)
    status, used, total_ms, cycles = struct.unpack_from("<bHII", pl, 0)
    rep = dict(status=status, windows=used, total_seconds=total_ms / 1000,
               compute_ms=round(cycles / MCU_CLOCK_HZ * 1e3, 2), anchor=a.anchor,
               scale=not a.no_scale, n_per_layer=a.n_per_layer)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if status != 0:
        print("校准未完成（状态码非 0）：-1 累加失败，-2 模型不含源域统计量，-3 被 STOP 中止")
    if a.dump and status == 0:
        n_layers = int(info.split("calib_layers=")[1].split()[0]) if "calib_layers=" in info else 7
        params = []
        for k in range(a.layers or n_layers):
            ser.write(make_frame(CMD_DUMP, bytes([k])))
            b, m, s = parse_dump(expect(ser, RSP_DUMP))
            params.append(dict(layer=k, bias=b, mult=m, shift=s))
        with open(a.dump, "w", encoding="utf-8") as f:
            json.dump(dict(report=rep, params=params), f, indent=1)
        print("参数已保存到", a.dump)
    ser.close()


if __name__ == "__main__":
    main()

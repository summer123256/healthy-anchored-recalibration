"""
开发板连通检查（烧录后第一件事）
==============================
电脑经串口问开发板“你是谁”（INFO），再让它用自带的测试向量做一次自检（SELFTEST）。
不需要数据集，也不需要导出目录。

用法：
  python board_check.py --port COM5
  （端口号在 Windows“设备管理器 → 端口(COM 和 LPT)”里看，CH340 那一项）
"""
import argparse
import time

from config import SERIAL_BAUD
from serial_proto import (CMD_INFO, CMD_SELFTEST, RSP_INFO, RSP_SELFTEST, expect, make_frame,
                          serial_open)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    a = ap.parse_args()
    ser = serial_open(a.port, a.baud, timeout=3)
    time.sleep(0.3)
    ser.reset_input_buffer()
    try:
        ser.write(make_frame(CMD_INFO))
        print("1) 开发板信息：", expect(ser, RSP_INFO).decode(errors="replace"))
        ser.write(make_frame(CMD_SELFTEST))
        pl = expect(ser, RSP_SELFTEST)
        ok, total = pl[0], pl[1]
        print(f"2) 片上自检：{ok}/{total} 个测试样本与电脑结果逐位一致")
        print("结论：", "全部正常，可以进行下一步" if ok == total and total > 0 else "自检没有全部通过，请把这段输出发给我")
    except Exception as e:                                   # noqa: BLE001
        print("没有收到开发板的正确回复：", e)
        print("请检查：TXD→A10、RXD→A9、GND→G 是否接好；端口号是否正确；程序是否已烧录。")
    finally:
        ser.close()


if __name__ == "__main__":
    main()

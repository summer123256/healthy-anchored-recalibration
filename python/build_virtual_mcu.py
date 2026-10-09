"""
编译“虚拟单片机”（没有硬件时测试电脑端串口脚本，仅 Linux / macOS / WSL，需要 gcc）
===========================================================================
把真实固件代码（app.c、tinycnn.c、adxl345.c、ssd1306.c）与导出的模型一起编译成电脑程序，
串口换成标准输入输出，ADXL345 换成模拟信号。

用法：
  python build_virtual_mcu.py --model-dir export/tiny
  python hil_test.py  --port virtual:export/tiny --model-dir export/tiny --n 50
  python hil_calib.py --port virtual:export/tiny --model-dir export/tiny --scenario C1
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FW = ROOT.parent / "firmware"


def build(model_dir):
    model_dir = Path(model_dir)
    app, host = FW / "app", FW / "host_test"
    exe = model_dir / "virtual_mcu"
    cmd = ["gcc", "-O2", "-std=gnu99", "-Wall", "-Wextra",
           f"-I{model_dir}", f"-I{host / 'mock'}", f"-I{app}", "-o", str(exe),
           str(host / "virtual_mcu.c"), str(app / "app.c"), str(app / "adxl345.c"), str(app / "ssd1306.c"),
           str(app / "tinycnn.c"), str(model_dir / "model_data.c"), "-lm"]
    subprocess.run(cmd, check=True)
    return exe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="export/tiny")
    a = ap.parse_args()
    print("已生成", build(a.model_dir))

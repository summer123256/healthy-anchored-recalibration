"""
自检脚本（不需要 PyTorch、不需要数据集）
=====================================
用随机权重构造与 models.py 结构相同的网络，走一遍
  量化 -> 整数参考推理 -> 导出 C -> gcc 编译 -> 逐位比对
用来确认 Python 端与 STM32 端 C 代码（推理 + 片上重校准）完全一致。

运行：python selftest.py        （需要系统里有 gcc）
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from export_c import (export_calib_vectors, export_model_c, export_vectors_bin,
                      export_vectors_header)
from preprocess import int8_to_model_input, preprocess_int16, to_int16
from quant import (conv_layer_indices, float_forward, fold_bn, int_forward, memory_report,
                   quantize)
from recal import WindowStream, recalibrate, source_stats

ROOT = Path(__file__).resolve().parent
FW = ROOT.parent / "firmware"


def _conv(rng, cin, cout, k, stride=1, pad=0, groups=1, relu=True):
    fan_in = cin // groups * k
    return dict(type="conv", w=rng.standard_normal((cout, cin // groups, k)) * np.sqrt(2 / fan_in),
                b=None, stride=stride, pad=pad, groups=groups, relu=relu,
                bn=dict(gamma=rng.uniform(0.5, 1.5, cout), beta=rng.normal(0, 0.1, cout),
                        mean=rng.normal(0, 0.05, cout), var=rng.uniform(0.5, 1.5, cout), eps=1e-5))


def _fc(rng, n_in, n_out, relu=False, bn=False):
    L = dict(type="fc", w=rng.standard_normal((n_out, n_in)) * np.sqrt(2 / n_in),
             b=rng.normal(0, 0.1, n_out), bn=None, relu=relu)
    if bn:
        L["bn"] = dict(gamma=rng.uniform(0.5, 1.5, n_out), beta=rng.normal(0, 0.1, n_out),
                       mean=rng.normal(0, 0.05, n_out), var=rng.uniform(0.5, 1.5, n_out), eps=1e-5)
    return L


def random_graph(name, rng, n_cls=10):
    mp = dict(type="maxpool", k=2)
    if name.startswith("tiny"):
        k1 = 128 if name == "tiny_k128" else 64
        sep = name != "tiny_std"
        g = [_conv(rng, 1, 16, k1, 8, (k1 - 8) // 2), mp]

        def block(ci, co):
            if sep:
                return [_conv(rng, ci, ci, 3, 1, 1, ci), _conv(rng, ci, co, 1)]
            return [_conv(rng, ci, co, 3, 1, 1)]
        g += block(16, 32) + [mp] + block(32, 32) + [mp] + block(32, 64)
        g += [dict(type="gap"), _fc(rng, 64, n_cls)]
        return g
    if name == "wdcnn":
        g = [_conv(rng, 1, 16, 64, 16, 24), mp, _conv(rng, 16, 32, 3, 1, 1), mp,
             _conv(rng, 32, 64, 3, 1, 1), mp, _conv(rng, 64, 64, 3, 1, 1), mp,
             _conv(rng, 64, 64, 3, 1, 0), mp, dict(type="flatten"),
             _fc(rng, 64, 100, relu=True, bn=True), _fc(rng, 100, n_cls)]
        return g
    raise ValueError(name)


def synthetic_vibration(rng, n, length=1024):
    """合成振动信号：若干正弦 + 周期冲击 + 噪声，幅值约几千个 int16 计数"""
    t = np.arange(length) / 12000.0
    out = np.empty((n, length))
    for i in range(n):
        f = rng.uniform(20, 300, 3)
        x = sum(rng.uniform(200, 2000) * np.sin(2 * np.pi * fi * t + rng.uniform(0, 6.3)) for fi in f)
        period = int(rng.uniform(60, 200))
        imp = np.zeros(length)
        imp[rng.integers(0, period)::period] = rng.uniform(2000, 12000)
        imp = np.convolve(imp, np.exp(-np.arange(40) / 6.0) * np.sin(np.arange(40) * 1.3), mode="same")
        out[i] = x + imp + rng.normal(0, 300, length) + rng.uniform(-500, 500)
    return out


EXE = ".exe" if os.name == "nt" else ""          # Windows（MinGW-w64）生成的程序带 .exe
RUN = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")


def gcc(out_exe, sources, includes):
    cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror"]
    cmd += [f"-I{p}" for p in includes] + ["-o", str(out_exe) + EXE] + [str(p) for p in sources]
    subprocess.run(cmd, check=True)


def exe(path):
    return str(path) + EXE


def run(name, n=300):
    rng = np.random.default_rng(1)
    graph = random_graph(name, rng)
    x16 = to_int16(synthetic_vibration(rng, n))
    y = rng.integers(0, 4, n)                     # 随机标签，仅用于计算“健康类”统计量
    xq = preprocess_int16(x16)
    xf = int8_to_model_input(xq)
    q = quantize(graph, xf[:200])
    logits = int_forward(q, xq[:, None, :])
    ref_float = float_forward(fold_bn(graph), xf.astype(np.float64))
    agree = np.mean(logits.argmax(1) == ref_float.argmax(1))
    stats = source_stats(q, xq, y)
    out = ROOT / "export" / f"selftest_{name}"
    export_model_c(q, out, [f"C{i}" for i in range(10)], model_name=name, stats=stats)
    export_vectors_bin(out / "vectors.bin", x16, logits)
    export_vectors_header(out / "test_vectors.h", x16[:2], logits[:2])
    app, host = FW / "app", FW / "host_test"

    # 1) 推理逐位比对
    gcc(out / "test_host", [host / "test_host.c", app / "tinycnn.c", out / "model_data.c"], [out, app])
    r = subprocess.run([exe(out / "test_host"), str(out / "vectors.bin")], **RUN)
    mem = memory_report(q)
    print(f"[{name}] 浮点与 int8 分类一致率 {agree:.3f} | 模型 Flash {mem['flash_model_bytes']} B | "
          f"激活 RAM {mem['ram_activation_bytes']} B")
    print("   " + r.stdout.strip().replace("\n", "\n   "))
    ok = r.returncode == 0

    # 2) 片上重校准逐位比对：构造一个“偏移后”的目标域，三种设置各测一次
    gcc(out / "test_calib", [host / "test_calib.c", app / "tinycnn.c", out / "model_data.c"], [out, app])
    n_layers = len(conv_layer_indices(q))
    per = 8
    tgt = synthetic_vibration(np.random.default_rng(7), per * n_layers + 40) * 0.6
    tgt += np.random.default_rng(8).normal(0, 400, tgt.shape)
    t16 = to_int16(tgt)
    cal16, test16 = t16[:per * n_layers], t16[per * n_layers:]
    for anchor, scale in (("healthy", True), ("all", True), ("healthy", False)):
        q2 = recalibrate(q, WindowStream(preprocess_int16(cal16)), stats, anchor, scale, per)
        expected = [(q2[li]["bq"], q2[li]["mult"], q2[li]["shift"]) for li in conv_layer_indices(q2)]
        tl = int_forward(q2, preprocess_int16(test16)[:, None, :])
        path = out / f"calib_{anchor}_{int(scale)}.bin"
        export_calib_vectors(path, [cal16[i * per:(i + 1) * per] for i in range(n_layers)], expected,
                             test16, tl, 0 if anchor == "healthy" else 1, int(scale))
        rc = subprocess.run([exe(out / "test_calib"), str(path)], **RUN)
        print("   " + rc.stdout.strip().replace("\n", "\n   "))
        ok = ok and rc.returncode == 0

    # 3) 串口协议、采集、片上校准流程仿真（模拟 HAL / UART / ADXL345）
    if name == "tiny":
        gcc(out / "sim_app", [host / "sim_app.c", app / "app.c", app / "adxl345.c", app / "ssd1306.c", app / "tinycnn.c",
                              out / "model_data.c"], [out, host / "mock", app])
        r2 = subprocess.run([exe(out / "sim_app"), str(out / "vectors.bin")], timeout=120, **RUN)
        print("   " + r2.stdout.strip().replace("\n", "\n   "))
        ok = ok and r2.returncode == 0
    return ok


def run_oled():
    """OLED 驱动：模拟 SSD1306 显存，检查初始化、普通字、大字、两种 I2C 地址、没有屏幕的情况"""
    host, app = FW / "host_test", FW / "app"
    out = ROOT / "export" / "selftest_oled"
    out.mkdir(parents=True, exist_ok=True)
    gcc(out / "test_oled", [host / "test_oled.c", app / "ssd1306.c"], [host / "mock_oled", app])
    r = subprocess.run([exe(out / "test_oled")], **RUN)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print(f"[oled] {last}")
    return r.returncode == 0


if __name__ == "__main__":
    if shutil.which("gcc") is None:
        print("没有找到 gcc，自检跳过。自检是可选的，不影响后面的实验。")
        print("Windows 安装方法：安装 MSYS2 后运行 "
              "pacman -S mingw-w64-ucrt-x86_64-gcc，并把 C:\\msys64\\ucrt64\\bin 加入 PATH。")
        sys.exit(0)
    names = sys.argv[1:] or ["tiny", "tiny_std", "tiny_k128", "wdcnn"]
    ok = all([run(n) for n in names] + [run_oled()])
    print("全部通过" if ok else "存在不一致！")
    sys.exit(0 if ok else 1)

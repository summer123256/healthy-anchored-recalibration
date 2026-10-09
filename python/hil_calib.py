"""
硬件在环片上重校准（用下载的公开数据，在 STM32 上真实执行重校准）
==============================================================
流程：
  1. 单片机恢复出厂参数（RESET）
  2. 逐层：电脑发送 N 个目标域“正常”窗口（CALIB_FEED），单片机累加统计量；
     然后单片机更新该层参数（CALIB_APPLY）
  3. 读取单片机上每层的新参数（DUMP），与 Python 的计算结果逐位比对
  4. 把目标域测试样本发给单片机推理，统计校准后的准确率，并与 Python 结果逐位比对

用法（默认 --layers 1，即论文方法 M3L1：只校准第一层）：
  python hil_calib.py --port COM5 --model-dir export/pu_tiny --scenario P1
  python hil_calib.py --port COM5 --model-dir export/pu_tiny --scenario P1 --layers 0   # 全部层（M3）
  python hil_calib.py --port COM5 --model-dir export/tiny --scenario C1
  python hil_calib.py --port COM5 --model-dir export/tiny_c2 --scenario C2-3HP --anchor all
  python hil_calib.py --port COM5 --model-dir export/fan_tiny --scenario F2
场景：C1（风扇端 FE 通道）；C2-1HP / C2-2HP / C2-3HP（驱动端）；P1 / P2 / P3（帕德博恩，用 pu_tiny 模型）；
      F1–F4（风扇台，可选）
注意：模型要与场景的源域一致（C1 用全负载 DE 训练的模型，C2 用 0 HP 训练的模型）。
"""
import argparse
import json
import pickle
import struct
import time
from pathlib import Path

import numpy as np

from config import MCU_CLOCK_HZ, SERIAL_BAUD
from preprocess import preprocess_int16, to_int16
from quant import conv_layer_indices, int_forward
from recal import WindowStream, recalibrate
from serial_proto import (CMD_CALIB_APPLY, CMD_CALIB_FEED, CMD_DUMP, CMD_INFER, CMD_INFO,
                          CMD_RESET_PARAMS, RSP_CALIB_APPLY, RSP_CALIB_FEED, RSP_DUMP, RSP_INFER,
                          RSP_INFO, RSP_RESET_PARAMS, expect, make_frame, parse_dump, parse_result,
                          serial_open)


def load_target(name):
    """返回 (正常状态校准窗口 X, 测试 X, 测试 y)，X 为浮点窗口（int16 计数单位）。"""
    if name == "C1" or name.startswith("C2-"):
        from data_cwru import build_cwru_target
        t = build_cwru_target((0, 1, 2, 3), "FE", synchronous=True) if name == "C1" else \
            build_cwru_target((int(name[3]),), "DE")
        Xc, yc, _ = t["calib"]
        Xt, yt, _ = t["test"]
        return Xc[yc == 0], Xt, yt
    if name in ("P1", "P2", "P3"):
        from data_pu import build_pu_target
        t = build_pu_target(name)
        Xc, yc, _ = t["calib"]
        Xt, yt, _ = t["test"]
        return Xc[yc == 0], Xt, yt
    from data_fan import build_fan_target
    t = build_fan_target(name)
    return t["calib_healthy"][0], t["test"][0], t["test"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=SERIAL_BAUD)
    ap.add_argument("--model-dir", default="export/tiny")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--anchor", default="healthy", choices=["healthy", "all"])
    ap.add_argument("--no-scale", action="store_true", help="只校正均值")
    ap.add_argument("--n-per-layer", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=300, help="发送给单片机的测试样本数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=1,
                    help="只校准前几个卷积层：1 = M3L1（论文方法，默认）；0 = 全部层（M3）")
    a = ap.parse_args()

    d = Path(a.model_dir)
    with open(d / "qmodel.pkl", "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "stats" not in obj:
        raise SystemExit("qmodel.pkl 中没有源域统计量，请用新版 quantize_export.py 重新导出")
    q, stats = obj["qlayers"], obj["stats"]
    n_cls = q[-1]["out_shape"][0]
    scale = not a.no_scale
    anchor_id = 0 if a.anchor == "healthy" else 1

    Xh, Xt, yt = load_target(a.scenario)
    rng = np.random.default_rng(a.seed)
    h16 = to_int16(Xh)[rng.permutation(len(Xh))]
    idx = rng.choice(len(yt), size=min(a.n_test, len(yt)), replace=False)
    t16, yt = to_int16(Xt[idx]), yt[idx]
    print(f"场景 {a.scenario}：正常校准窗口 {len(h16)} 个，测试样本 {len(yt)} 个")

    # Python 端期望结果
    layers = conv_layer_indices(q)
    if a.layers:
        layers = layers[:a.layers]
    q2 = recalibrate(q, WindowStream(preprocess_int16(h16)), stats, a.anchor, scale, a.n_per_layer,
                     n_layers=len(layers))
    print(f"校准层数：{len(layers)}（{'M3L1' if len(layers) == 1 and a.anchor == 'healthy' else '对比设置'}）")
    ref_before = int_forward(q, preprocess_int16(t16)[:, None, :])
    ref_after = int_forward(q2, preprocess_int16(t16)[:, None, :])

    ser = serial_open(a.port, a.baud, timeout=5)
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(make_frame(CMD_INFO))
    print("单片机信息:", expect(ser, RSP_INFO).decode(errors="replace"))
    ser.write(make_frame(CMD_RESET_PARAMS))
    expect(ser, RSP_RESET_PARAMS)

    t0 = time.time()
    apply_cycles, param_ok, feed_cycles = [], [], {}
    ch_ok, ch_total = 0, 0
    pos = 0
    for k, li in enumerate(layers):
        for _ in range(a.n_per_layer):
            w = h16[pos % len(h16)]
            pos += 1
            ser.write(make_frame(CMD_CALIB_FEED, bytes([k]) + w.astype("<i2").tobytes()))
            pl = expect(ser, RSP_CALIB_FEED)
            st, _, fcyc = struct.unpack_from("<bHI", pl, 0)
            if st != 0:
                raise RuntimeError(f"第 {k} 层累加失败")
            feed_cycles[k] = feed_cycles.get(k, 0) + fcyc
        ser.write(make_frame(CMD_CALIB_APPLY, bytes([k, anchor_id, int(scale)])))
        pl = expect(ser, RSP_CALIB_APPLY)
        status, cyc = struct.unpack_from("<bI", pl, 0)
        if status != 0:
            raise RuntimeError(f"第 {k} 层更新失败（状态 {status}），模型是否包含源域统计量？")
        apply_cycles.append(cyc)
        ser.write(make_frame(CMD_DUMP, bytes([k])))
        b, m, s = parse_dump(expect(ser, RSP_DUMP))
        rb = [int(v) for v in q2[li]["bq"]]
        rm = [int(v) for v in q2[li]["mult"]]
        rs = [int(v) for v in q2[li]["shift"]]
        n_same = sum(int(b[c] == rb[c] and m[c] == rm[c] and s[c] == rs[c]) for c in range(len(rb)))
        ch_ok += n_same
        ch_total += len(rb)
        same = n_same == len(rb)
        param_ok.append(same)
        print(f"  第 {k + 1} 层：{n_same}/{len(rb)} 个通道的 b、M、s 与 Python 一致{'' if same else '  ← 不一致！'}")

    # 恢复出厂参数后先测校准前的单片机输出（论文 5.7 节“校准前后 logits 逐位一致”）
    ser.write(make_frame(CMD_RESET_PARAMS))
    expect(ser, RSP_RESET_PARAMS)
    preds0, exact0 = [], 0
    for i in range(len(yt)):
        ser.write(make_frame(CMD_INFER, t16[i].astype("<i2").tobytes()))
        pred, _, _, logits = parse_result(expect(ser, RSP_INFER), n_cls)
        preds0.append(pred)
        exact0 += int(logits == [int(v) for v in ref_before[i]])
    # 再次按同样的窗口校准，然后测校准后的输出
    ser.write(make_frame(CMD_RESET_PARAMS))
    expect(ser, RSP_RESET_PARAMS)
    pos = 0
    for k, li in enumerate(layers):
        for _ in range(a.n_per_layer):
            w = h16[pos % len(h16)]
            pos += 1
            ser.write(make_frame(CMD_CALIB_FEED, bytes([k]) + w.astype("<i2").tobytes()))
            expect(ser, RSP_CALIB_FEED)
        ser.write(make_frame(CMD_CALIB_APPLY, bytes([k, anchor_id, int(scale)])))
        expect(ser, RSP_CALIB_APPLY)

    preds, exact = [], 0
    for i in range(len(yt)):
        ser.write(make_frame(CMD_INFER, t16[i].astype("<i2").tobytes()))
        pred, _, _, logits = parse_result(expect(ser, RSP_INFER), n_cls)
        preds.append(pred)
        exact += int(logits == [int(v) for v in ref_after[i]])
    ser.close()

    preds = np.array(preds)
    preds0 = np.array(preds0)
    rep = dict(scenario=a.scenario, anchor=a.anchor, scale=scale, n_per_layer=a.n_per_layer,
               n_layers=len(layers), n_test=int(len(yt)),
               acc_before_mcu=float((preds0 == yt).mean()),
               logits_bit_exact_rate_before=exact0 / len(yt),
               params_bit_exact_channels=f"{ch_ok}/{ch_total}",
               acc_before_python=float((ref_before.argmax(1) == yt).mean()),
               acc_after_python=float((ref_after.argmax(1) == yt).mean()),
               acc_after_mcu=float((preds == yt).mean()),
               params_bit_exact_layers=f"{sum(param_ok)}/{len(param_ok)}",
               logits_bit_exact_rate=exact / len(yt),
               apply_ms_per_layer=[round(c / MCU_CLOCK_HZ * 1e3, 3) for c in apply_cycles],
               feed_ms_per_layer=[round(feed_cycles[k] / MCU_CLOCK_HZ * 1e3, 2) for k in sorted(feed_cycles)],
               # 片上计算总时间（不含串口传输；现场采集时再加上 N×层数 个窗口的采集时间）
               calib_compute_ms_total=round((sum(apply_cycles) + sum(feed_cycles.values()))
                                            / MCU_CLOCK_HZ * 1e3, 2),
               wall_seconds=round(time.time() - t0, 1))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    tag = "L1" if len(layers) == 1 else f"L{len(layers)}"
    out = d / f"hil_calib_{a.scenario}_{a.anchor}_{tag}{'' if scale else '_meanonly'}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("已保存", out)


if __name__ == "__main__":
    main()

"""
把量化后的模型导出为 STM32 可直接编译的 C 文件
============================================
生成：
  model_data.h / model_data.c  —— 网络结构与 int8 权重（放进 firmware/app/）
  test_vectors.h               —— 片上自检用的少量测试向量（放进 firmware/app/）
  vectors.bin                  —— 电脑端 gcc 验证用的大量测试向量（firmware/host_test/）
"""
import struct
from pathlib import Path

import numpy as np

TYPE_ID = {"conv": 0, "fc": 1, "maxpool": 2, "gap": 3, "flatten": 4}


def _arr(ctype, name, a, per_line=16):
    a = np.asarray(a).ravel()
    body = []
    for i in range(0, len(a), per_line):
        body.append("    " + ", ".join(str(int(v)) for v in a[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(a)}] = {{\n" + "\n".join(body) + "\n};\n"


def export_model_c(qlayers, out_dir, class_names, model_name="tiny", input_len=1024, stats=None):
    """stats: recal.source_stats() 的结果（可选）。提供后，片上重校准才可用。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, entries = [], []
    max_act = input_len
    calib_layers, ch_off, max_cal_ch = [], 0, 1
    for i, L in enumerate(qlayers):
        t = L["type"]
        cin, lin = L["in_shape"]
        cout, lout = L["out_shape"]
        max_act = max(max_act, cin * lin, cout * lout)
        w = b = m = s = "0"
        mh = sh = ma = sa = "0"
        k = st = pad = g = 0
        off = 0
        if t in ("conv", "fc"):
            arrays.append(_arr("int8_t", f"w{i}", L["wq"]))
            arrays.append(_arr("int32_t", f"b{i}", L["bq"]))
            w, b = f"w{i}", f"b{i}"
            if not L["last"]:
                arrays.append(_arr("int32_t", f"m{i}", L["mult"]))
                arrays.append(_arr("int8_t", f"s{i}", L["shift"]))
                m, s = f"m{i}", f"s{i}"
            if t == "conv":
                k, st, pad, g = L["wq"].shape[2], L["stride"], L["pad"], L["groups"]
                if not L["last"]:
                    calib_layers.append(i)
                    off = ch_off
                    ch_off += cout
                    max_cal_ch = max(max_cal_ch, cout)
                    if stats is not None:
                        for tag in ("mu_h", "sig_h", "mu_a", "sig_a"):
                            arrays.append(_arr("int32_t", f"{tag}{i}", stats[i][tag]))
                        mh, sh, ma, sa = f"mu_h{i}", f"sig_h{i}", f"mu_a{i}", f"sig_a{i}"
            else:
                cin, lin = cin * lin, 1
        elif t == "maxpool":
            k = st = L["k"]
        entries.append(
            f"    {{ {TYPE_ID[t]}, {int(L.get('relu', 0))}, {int(L.get('last', 0))}, "
            f"{cin}, {lin}, {cout}, {lout}, {k}, {st}, {pad}, {g}, {off}, {w}, {b}, {m}, {s}, "
            f"{mh}, {sh}, {ma}, {sa} }},  /* {i}: {t} */")
    n_classes = qlayers[-1]["out_shape"][0]
    n_cal = len(calib_layers)
    h = f"""/* 自动生成，请勿手工修改：python quantize_export.py */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include "tinycnn.h"

#define TC_MODEL_NAME    "{model_name}"
#define TC_INPUT_LEN     {input_len}
#define TC_NUM_CLASSES   {n_classes}
#define TC_NUM_LAYERS    {len(qlayers)}
#define TC_MAX_ACT       {max_act}   /* 单个激活缓冲区字节数（共用两个，乒乓） */
#define TC_NUM_CALIB     {n_cal}   /* 可校准卷积层个数 */
#define TC_CALIB_CH      {max(ch_off, 1)}   /* 可校准通道总数 */
#define TC_MAX_CALIB_CH  {max_cal_ch}   /* 单层最大通道数 */
#define TC_HAS_STATS     {1 if stats is not None else 0}   /* 是否包含源域统计量 */

extern const tc_layer_t tc_layers[TC_NUM_LAYERS];
extern const uint8_t tc_calib_layers[{max(n_cal, 1)}];
extern const char *const tc_class_names[TC_NUM_CLASSES];

#endif
"""
    names = ", ".join(f'"{c}"' for c in class_names)
    cal = ", ".join(str(i) for i in calib_layers) or "0"
    c = ('/* 自动生成，请勿手工修改：python quantize_export.py */\n'
         '#include "model_data.h"\n\n' + "\n".join(arrays) +
         "\nconst tc_layer_t tc_layers[TC_NUM_LAYERS] = {\n"
         "  /* type relu last in_ch in_len out_ch out_len k stride pad groups ch_off w b mult shift"
         " mu_h sig_h mu_a sig_a */\n"
         + "\n".join(entries) + "\n};\n\n"
         f"const uint8_t tc_calib_layers[{max(n_cal, 1)}] = {{ {cal} }};\n"
         f"const char *const tc_class_names[TC_NUM_CLASSES] = {{ {names} }};\n")
    (out_dir / "model_data.h").write_text(h, encoding="utf-8")
    (out_dir / "model_data.c").write_text(c, encoding="utf-8")
    return max_act


def export_vectors_bin(path, x16, logits):
    """格式：'TCV1' | n | input_len | n_classes | 然后每条：int16[input_len] int32[n_classes]"""
    x16 = np.asarray(x16, dtype="<i2")
    logits = np.asarray(logits, dtype="<i4")
    with open(path, "wb") as f:
        f.write(b"TCV1" + struct.pack("<III", len(x16), x16.shape[1], logits.shape[1]))
        for a, b in zip(x16, logits):
            f.write(a.tobytes())
            f.write(b.tobytes())


def export_vectors_header(path, x16, logits):
    x16 = np.asarray(x16)
    logits = np.asarray(logits)
    txt = ["/* 自动生成：片上自检测试向量 */", "#ifndef TEST_VECTORS_H", "#define TEST_VECTORS_H",
           "#include <stdint.h>", f"#define TV_COUNT {len(x16)}", ""]
    txt.append(f"static const int16_t tv_input[TV_COUNT][{x16.shape[1]}] = {{")
    for row in x16:
        txt.append("  {" + ", ".join(str(int(v)) for v in row) + "},")
    txt.append("};")
    txt.append(f"static const int32_t tv_logits[TV_COUNT][{logits.shape[1]}] = {{")
    for row in logits:
        txt.append("  {" + ", ".join(str(int(v)) for v in row) + "},")
    txt += ["};", "#endif", ""]
    Path(path).write_text("\n".join(txt), encoding="utf-8")


def export_calib_vectors(path, windows16, expected, test16, test_logits, anchor, scale):
    """片上重校准的逐位比对向量（电脑端 gcc 测试用）。
    windows16: 列表，每个可校准层一个 (N, WIN) int16 数组（按层顺序使用）
    expected : 列表，每层 (b, mult, shift) 三个数组（校准该层之后的值）
    格式：'TCC1' | anchor | scale | 层数 | N | WIN | 每层：N 个窗口 + 通道数 + b + mult + shift
          | 测试条数 | 类别数 | 每条：int16[WIN] + int32 logits"""
    n_layers = len(windows16)
    n = windows16[0].shape[0]
    win = windows16[0].shape[1]
    with open(path, "wb") as f:
        f.write(b"TCC1" + struct.pack("<IIIII", anchor, scale, n_layers, n, win))
        for w, (b, m, s) in zip(windows16, expected):
            f.write(np.asarray(w, dtype="<i2").tobytes())
            f.write(struct.pack("<I", len(b)))
            f.write(np.asarray(b, dtype="<i4").tobytes())
            f.write(np.asarray(m, dtype="<i4").tobytes())
            f.write(np.asarray(s, dtype="<i4").tobytes())
        test16 = np.asarray(test16, dtype="<i2")
        test_logits = np.asarray(test_logits, dtype="<i4")
        f.write(struct.pack("<II", len(test16), test_logits.shape[1]))
        for a, b in zip(test16, test_logits):
            f.write(a.tobytes())
            f.write(b.tobytes())

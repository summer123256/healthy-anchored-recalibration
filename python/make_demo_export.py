"""
生成一个“演示用”导出目录（随机权重 + 合成振动信号，不需要 PyTorch、不需要数据集）
=========================================================================
用途：在还没有训练好的模型之前，先测试 烧录 / 串口 / OLED / 耗时，或者配合“虚拟单片机”
测试全部串口脚本。它的准确率没有任何意义，不能用于论文。

用法：
  python make_demo_export.py                 # 输出到 export/demo_tiny/
  python build_virtual_mcu.py --model-dir export/demo_tiny      # （Linux / macOS / WSL）
  python hil_test.py --port virtual:export/demo_tiny --model-dir export/demo_tiny --n 20
烧录到单片机时，把 export/demo_tiny/ 里的 model_data.c、model_data.h、test_vectors.h 复制到工程。
"""
import argparse
import pickle

import numpy as np

from config import EXPORT_DIR
from export_c import export_model_c, export_vectors_bin, export_vectors_header
from preprocess import int8_to_model_input, preprocess_int16, to_int16
from quant import int_forward, quantize
from recal import source_stats
from selftest import random_graph, synthetic_vibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="demo_tiny")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    classes = [f"C{i}" for i in range(10)]
    graph = random_graph("tiny", rng, n_cls=len(classes))
    x_cal = synthetic_vibration(rng, 200)
    q = quantize(graph, int8_to_model_input(preprocess_int16(to_int16(x_cal))))
    y_cal = rng.integers(0, len(classes), len(x_cal))
    y_cal[:40] = 0                                   # 保证有“正常类”样本，用于源域统计量
    stats = source_stats(q, preprocess_int16(to_int16(x_cal)), y_cal, healthy_label=0)

    x16 = to_int16(synthetic_vibration(rng, 100))
    logits = int_forward(q, preprocess_int16(x16)[:, None, :])
    y = logits.argmax(1)                             # 以模型自己的预测为“标签”，仅用于测试流程

    out = EXPORT_DIR / a.name
    export_model_c(q, out, classes, model_name=a.name, stats=stats)
    export_vectors_bin(out / "vectors.bin", x16, logits)
    export_vectors_header(out / "test_vectors.h", x16[:2], logits[:2])
    with open(out / "qmodel.pkl", "wb") as f:
        pickle.dump(dict(qlayers=q, stats=stats, classes=classes), f)
    np.savez_compressed(out / "hil_testset.npz", x16=x16, y=y, classes=np.array(classes))
    print(f"已生成 {out}（随机权重，仅用于测试流程，准确率没有意义）")


if __name__ == "__main__":
    main()

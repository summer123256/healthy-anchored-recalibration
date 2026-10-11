"""
量化 + 导出 C 代码
=================
用法：
  python quantize_export.py --ckpt checkpoints/cwru_tiny_main.pt --name tiny
  python quantize_export.py --ckpt checkpoints/pu_tiny_s0.pt --name pu_tiny      （帕德博恩数据）
  python quantize_export.py --ckpt checkpoints/tta_J_s0_e60.pt --name jnu_tiny --dataset jnu   （江南大学数据）
  （CWRU 跨负载场景 C2 只用 0 HP 训练时，加 --train-loads 0）

步骤：
  1. 读取训练好的 PyTorch 模型
  2. 合并 BN，用训练集样本校准激活范围，int8 量化
  3. 对比 PyTorch 浮点 / numpy 浮点 / int8 整数 三者的测试集准确率
  4. 输出到 export/<name>/：
       model_data.h, model_data.c   -> 复制到 STM32 工程（含源域统计量，支持片上重校准）
       test_vectors.h               -> 复制到 STM32 工程（片上自检）
       vectors.bin                  -> 电脑端 gcc 验证
       qmodel.pkl, hil_testset.npz  -> 硬件在环测试 (hil_test.py) 使用
       report.json                  -> 存储占用、准确率等
"""
import argparse
import json
import pickle

import numpy as np

from config import EXPORT_DIR, JNU_CLASSES, MCU_FLASH_BYTES, MCU_RAM_BYTES
from export_c import export_model_c, export_vectors_bin, export_vectors_header
from preprocess import int8_to_model_input, preprocess_int16, to_int16
from quant import float_forward, fold_bn, int_forward, memory_report, quantize
from recal import source_stats


def infer_dataset(classes, n_classes):
    """模型文件里没有记录数据集时，按类别推断：JNU 的 4 类、帕德博恩的 3 类，其余为 CWRU。"""
    if list(classes) == list(JNU_CLASSES):
        return "jnu"
    return "pu" if n_classes == 3 else "cwru"


def load_source(dataset, channel="DE", loads=(0, 1, 2, 3)):
    """读取源域数据：{'train'|'val'|'test': (X, y, 类别)}，X 为浮点窗口（int16 计数单位）。"""
    if dataset == "cwru":
        from data_cwru import build_cwru
        return build_cwru(loads=tuple(loads), channel=channel)
    if dataset == "pu":
        from data_pu import build_pu_source
        return build_pu_source()
    if dataset == "jnu":
        from data_jnu import build_jnu_source
        return build_jnu_source()
    from data_fan import build_fan_source
    return build_fan_source()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="tiny")
    ap.add_argument("--dataset", default=None, choices=["cwru", "pu", "jnu", "fan"],
                    help="默认从模型文件中读取训练时的设置")
    ap.add_argument("--channel", default=None, choices=["DE", "FE"])
    ap.add_argument("--train-loads", type=int, nargs="+", default=None,
                    help="源域负载（须与训练时一致，用于量化校准与源域统计量）")
    ap.add_argument("--calib", type=int, default=1000, help="校准样本数")
    ap.add_argument("--act-percentile", type=float, default=None, help="如 99.99；默认用最大值")
    ap.add_argument("--target-vectors", type=int, default=2, help="写入单片机的自检向量数")
    a = ap.parse_args()

    import torch
    from models import build_model, export_graph
    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(ck["model"], ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    classes = ck["classes"]
    cfg = ck.get("args", {}) or {}
    dataset = a.dataset or cfg.get("dataset")
    if dataset is None:   # 模型文件里没有记录数据集时按类别推断（建议显式给出 --dataset）
        dataset = infer_dataset(classes, ck["n_classes"])
        print(f"提示：模型文件未记录数据集，按类别推断为 {dataset}；如不对请加 --dataset")
    channel = a.channel or cfg.get("channel", "DE")
    loads = a.train_loads or cfg.get("train_loads", [0, 1, 2, 3])
    print(f"源域设置：dataset={dataset}" + (f"，channel={channel}，负载={loads}" if dataset == "cwru" else ""))

    ds = load_source(dataset, channel, loads)
    Xtr, ytr, _ = ds["train"]
    Xte, yte, _ = ds["test"]

    rng = np.random.default_rng(0)
    cal_idx = rng.choice(len(Xtr), size=min(a.calib, len(Xtr)), replace=False)
    xq_cal = preprocess_int16(to_int16(Xtr[cal_idx]))
    x16_te = to_int16(Xte)
    xq_te = preprocess_int16(x16_te)
    xf_te = int8_to_model_input(xq_te)

    graph = export_graph(model)
    q = quantize(graph, int8_to_model_input(xq_cal), a.act_percentile)
    # 源域统计量（健康类与全部类别），供片上重校准使用
    st_idx = rng.choice(len(Xtr), size=min(3000, len(Xtr)), replace=False)
    stats = source_stats(q, preprocess_int16(to_int16(Xtr[st_idx])), ytr[st_idx], healthy_label=0)

    with torch.no_grad():
        logit_torch = torch.cat([model(torch.from_numpy(xf_te[i:i + 512]))
                                 for i in range(0, len(xf_te), 512)]).numpy()
    logit_np = np.concatenate([float_forward(fold_bn(graph), xf_te[i:i + 512].astype(np.float64))
                               for i in range(0, len(xf_te), 512)])
    logit_int = int_forward(q, xq_te[:, None, :])
    diff = float(np.abs(logit_torch - logit_np).max())
    acc_t = float((logit_torch.argmax(1) == yte).mean())
    acc_i = float((logit_int.argmax(1) == yte).mean())
    print(f"PyTorch 与 numpy 浮点 logits 最大差 {diff:.2e}（应接近 0，否则说明导出有误）")
    print(f"测试准确率：浮点 {acc_t:.4f}，int8 {acc_i:.4f}，下降 {100 * (acc_t - acc_i):.2f} 个百分点")

    out = EXPORT_DIR / a.name
    max_act = export_model_c(q, out, classes, model_name=a.name, stats=stats)
    export_vectors_bin(out / "vectors.bin", x16_te, logit_int)
    tv = rng.choice(len(x16_te), size=a.target_vectors, replace=False)
    export_vectors_header(out / "test_vectors.h", x16_te[tv], logit_int[tv])
    with open(out / "qmodel.pkl", "wb") as f:
        pickle.dump(dict(qlayers=q, stats=stats, classes=list(classes)), f)
    np.savez_compressed(out / "hil_testset.npz", x16=x16_te, y=yte, classes=np.array(classes))

    mem = memory_report(q)
    rep = dict(name=a.name, ckpt=a.ckpt, acc_float=acc_t, acc_int8=acc_i, torch_numpy_maxdiff=diff,
               n_test=int(len(yte)), **mem, tc_max_act=max_act,
               ram_input_bytes=1024 * 2 + 1024,  # int16 原始窗口 + int8 预处理结果
               ram_calib_bytes=9 * sum(L["out_shape"][0] for L in q if L["type"] == "conv" and not L["last"])
               + 16 * max(L["out_shape"][0] for L in q if L["type"] == "conv") + 4,   # +4：共用计数器
               flash_budget=MCU_FLASH_BYTES, ram_budget=MCU_RAM_BYTES)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(mem, indent=2))
    ram = mem["ram_activation_bytes"] + rep["ram_input_bytes"]
    if ram > MCU_RAM_BYTES * 0.7:
        print(f"警告：推理所需 RAM 约 {ram} 字节，超过 20 KB 的 70%，在 STM32F103C8T6 上可能放不下！")
    if mem["flash_model_bytes"] > MCU_FLASH_BYTES * 0.6:
        print("警告：模型权重超过 Flash 的 60%，加上程序代码可能放不下！")
    print(f"已导出到 {out}")


if __name__ == "__main__":
    main()

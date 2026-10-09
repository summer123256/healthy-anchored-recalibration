"""
汇总单片机实测结果（论文表 11：STM32F103C8T6 上的资源与时间）
==========================================================
读取：
  export/<模型>/report.json        模型 Flash、RAM 估算（quantize_export.py 生成）
  export/<模型>/hil_report.json    推理耗时、逐位一致率（hil_test.py 生成）
  export/<模型>/hil_calib_*.json   片上校准耗时、校准前后准确率（hil_calib.py 生成）
以及（可选）firmware 编译输出的 .map / size 结果由你手工填写（见说明文档第 8 节）。

用法：python hil_summary.py
输出：results/table11_mcu.csv 以及屏幕打印（缺少的文件会提示）
"""
import csv
import json

from config import EXPORT_DIR, RESULT_DIR


def jload(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    rows = []
    for d in sorted(p for p in EXPORT_DIR.glob("*") if p.is_dir()):
        rep, hil = jload(d / "report.json"), jload(d / "hil_report.json")
        if rep is None:
            continue
        row = dict(model=d.name, flash_model_bytes=rep.get("flash_model_bytes"),
                   ram_activation_bytes=rep.get("ram_activation_bytes"), ram_calib_bytes=rep.get("ram_calib_bytes"),
                   acc_int8_pc=rep.get("acc_int8"))
        if hil:
            for k in ("pre_ms_mean", "infer_ms_mean", "infer_ms_max", "total_ms_mean", "bit_exact_rate", "acc_mcu", "n"):
                if k in hil:
                    row[k] = hil[k]
        else:
            print(f"提示：{d.name} 没有 hil_report.json（运行 hil_test.py 后生成）")
        rows.append(row)
    for p in sorted(EXPORT_DIR.glob("*/hil_calib_*.json")):
        r = jload(p)
        if r:
            rows.append(dict(model=f"{p.parent.name}/{p.stem}", scenario=r.get("scenario"), anchor=r.get("anchor"),
                             acc_before=r.get("acc_before_python"), acc_after_mcu=r.get("acc_after_mcu"),
                             acc_after_python=r.get("acc_after_python"),
                             n_layers=r.get("n_layers"),
                             acc_before_mcu=r.get("acc_before_mcu"),
                             params_bit_exact=r.get("params_bit_exact_channels", r.get("params_bit_exact_layers")),
                             logits_bit_exact_rate_before=r.get("logits_bit_exact_rate_before"),
                             logits_bit_exact_rate=r.get("logits_bit_exact_rate"),
                             calib_compute_ms_total=r.get("calib_compute_ms_total"),
                             apply_ms_per_layer=r.get("apply_ms_per_layer")))
    if not rows:
        raise SystemExit("没有找到任何结果文件")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = RESULT_DIR / "table11_mcu.csv"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()

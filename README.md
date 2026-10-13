# Healthy-anchored on-device recalibration on a 20 KB-RAM MCU

Code for the paper **"Healthy-Data-Only, Gradient-Free On-Device Recalibration of an Integer-Only Vibration Fault Classifier on a 20 KB-RAM Microcontroller"**.

The supplementary material of the paper (synthetic-data validation, α sensitivity of α-BN, macro-F1, additional figures, the JNU check, per-target statistics, gain limits and confusion matrices) is in [`docs/Supplementary_material.pdf`](docs/Supplementary_material.pdf).

A tiny 1-D CNN is trained on bearing vibration data, quantised to int8 and run on an STM32F103C8T6 (72 MHz Cortex-M3, 20 KB RAM, no FPU). After a domain shift, the board recalibrates the first convolutional layer using only 32 healthy-condition windows: per-channel accumulator statistics are aligned to healthy-class anchors computed offline, and the correction is folded into the integer bias and requantisation multiplier (method **M3L1**). No labels, no gradients, no floating point.

## Repository layout

| Folder | Content |
|---|---|
| `python/` | Data loading (CWRU, Paderborn), training, int8 quantisation, recalibration methods (M0–M4, B1–B3), all experiments, statistics, figures, C export, hardware-in-the-loop (HIL) scripts |
| `firmware/app/` | Integer-only pre-processing, inference and on-board recalibration (`tinycnn.c`), UART protocol (`app.c`), optional OLED / ADXL345 drivers |
| `firmware/project_make/` | Complete STM32 HAL project for the STM32F103C8T6 (`CMakeLists.txt` and `Makefile`), linker script, OpenOCD config |
| `firmware/model/` | Placeholder model (random weights) for testing the toolchain; replaced by an exported model |
| `firmware/host_test/` | PC builds of the firmware: bit-exactness tests and a "virtual MCU" for testing the serial scripts without hardware |

## 1. Computer experiments (Sections 4–5.4 of the paper and the supplementary material)

Python 3.10+:

```bash
cd python
python -m venv venv && venv\Scripts\activate        # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For exactly the package versions used in the paper (Python 3.11.9, PyTorch 2.14.1 CPU build, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1), install from the lock file instead:

```bash
pip install -r requirements_lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The paper's computer experiments ran on the CPU of a laptop (AMD Ryzen 7 4800H, 16 GB RAM, Windows 10).

Data (not included):

* **CWRU**: `python download_cwru.py` downloads the required `.mat` records (12 kHz drive-end faults and the normal baseline) to `data/cwru/`.
* **Paderborn (PU)**: download `K001, K002, KA04, KA15, KI04, KI14` (`.rar`) from the KAt Bearing DataCenter of Paderborn University and extract them to `data/pu/<bearing>/`. Check with `python data_pu.py`.

Run (this is the exact sequence used for the paper; `--use-ckpt` reuses the trained source models, and because training is deterministic it gives the same results as retraining):

```bash
python check_theory.py                                   # Supplementary S1: Table S1, Fig. S1 (synthetic data)
python experiments_tta.py --scenario C1                  # E0, E1, E5: main results, 5 seeds, 60 epochs
python experiments_tta.py --scenario C2
python experiments_tta.py --scenario P
python experiments_tta.py --scenario C1 --ablation-only --use-ckpt   # E2-E4 ablations and E5 for every seed
python experiments_tta.py --scenario C2 --ablation-only --use-ckpt
python experiments_tta.py --scenario P  --ablation-only --use-ckpt
python experiments_tta.py --scenario C1 --supplement --use-ckpt      # M3L1 in E0, alpha sweep of alpha-BN
python experiments_tta.py --scenario C2 --supplement --use-ckpt
python experiments_tta.py --scenario P  --supplement --use-ckpt
python experiments_tta.py --gate                         # pre-registered success criteria
python b2_tuned_report.py                                # selects alpha for B2* on the development targets
python stats_test.py                                     # Wilcoxon + Holm, all 7 targets (Table 7, top)
python stats_test.py --held-out                          # held-out targets only (Table 7, bottom)
python stats_test.py --ref M3                            # statistics for M3
python make_figures.py --paper --lang en                 # Fig. 3 and Figs. S1-S5 (use --lang zh for Chinese labels)
python run_m1l1.py --use-ckpt                            # M1L1 control: first layer only, all-class anchor
python -X utf8 clip_and_confusion.py                     # gain limits of M3L1 and confusion matrices (Section S8), saved models only
```

Check on a third dataset (Section 5.2, finding (5); supplementary S6): the Jiangnan University (JNU) bearing data, speed change, with all settings and the decision rule (M3L1 above M0 on both targets) fixed before any result was seen: source 1000 r/min, targets J1 = 800 r/min and J2 = 600 r/min. Put the 12 `.csv` files of https://github.com/ClarkGableWang/JNU-Bearing-Dataset in `data/jnu/`, then:

```bash
python data_jnu.py                                       # check files and window counts
python experiments_tta.py --scenario J                   # main results, 5 seeds, 60 epochs
python experiments_tta.py --scenario J --supplement --use-ckpt   # B2* (alpha = 0.1)
python run_m1l1.py --use-ckpt --scenarios J              # M1L1 control -> tta_m1l1_J*.csv
```

On Windows, when the output is redirected to a file (`> log.txt`), add `-X utf8` (for example `python -X utf8 experiments_tta.py --scenario J > J_log.txt 2>&1`), because some printed symbols cannot be encoded in the default console code page.

Where each result appears in the paper:

| Result file (`results/`) | Paper |
|---|---|
| `theory_check.json` | Table S1, Fig. S1 (`fig3_theory_synthetic`) |
| `tta_<scenario>_e0.csv`, `tta_<scenario>_supp.csv` | Table 5 |
| `tta_<scenario>_main.csv`, `tta_<scenario>_summary.csv` | Table 6, Table S3, Fig. S2 (`fig4_main`) |
| `stats_wilcoxon_M3L1.csv`, `stats_wilcoxon_M3L1_heldout.csv` | Table 7 |
| `tta_<scenario>_supp.csv`, `b2_tuned_report.json` | Table S2 and B2\* in Tables 6–7 |
| `tta_<scenario>_ablation.csv` | Table 8, Fig. 3 (`fig5_contamination`), Figs. S3 (`fig8_layers`) and S4 (`fig6_windows`) |
| `tta_<scenario>_e5_allseeds.json` | Table 9, Fig. S5 (`fig7_assumption_a`) |
| `tta_m1l1.csv`, `tta_m1l1_e0.csv`, `tta_m1l1_check.json` | M1L1 in Section 5.1–5.2, Table 8, Table S4 |
| `tta_J_*.csv`, `tta_J_source.json`, `tta_m1l1_J*.csv`, `tta_m1l1_J_check.json` | Section 5.2 (5), Table S5 and Section S6 |
| `clip_confusion/clip_summary.txt`, `clip_confusion/*.csv`, `clip_confusion/confusion_*.png` | Section 3.3 (gain limits), Section 5.2, Section S8, Table S7, Fig. S6 |
| build output, `table11_mcu.csv`, `export/<model>/hil_*.json` | Tables 10 and 11 |

The source models are saved as `checkpoints/tta_<scenario>_s<seed>_e60.pt`.

`run_m1l1.py` also recomputes M3L1 with the same calibration windows and checks it against `tta_<scenario>_main.csv` (`tta_m1l1_check.json`). Every on-board calibration starts from the factory parameters, so repeated calibrations do not accumulate.

## 2. On-board experiment (Section 5.5, Tables 10–11)

Hardware: STM32F103C8T6 minimum system board, ST-Link V2, USB-to-UART module (CH340). Wiring:

| From | To board pin |
|---|---|
| ST-Link SWDIO / SWCLK / GND / 3.3V | DIO / CLK / GND / 3.3 (SWD header) |
| UART TXD | A10 (USART1 RX) |
| UART RXD | A9 (USART1 TX) |
| UART GND | G |

UART: 460 800 bit/s, 8N1. OLED (I2C1, PB6/PB7) and ADXL345 are optional and not used in the paper.

### Exploratory code not used in the paper

The repository also contains an optional pipeline for a self-built fan test rig with an ADXL345 accelerometer (3200 Hz): `python/fan_collect.py`, `python/data_fan.py`, `python/device_calibrate.py`, `python/online_monitor.py`, the optional scenario `F` hooks in `experiments_tta.py`, `train.py`, `quantize_export.py`, `hil_calib.py` and `config.py`, and the ADXL345 / OLED drivers in `firmware/app/`. It was written for possible future work; no fan data were collected, and no result in the paper or the supplementary material depends on it. `python/experiments.py` and `python/baseline_svm.py` belong to an earlier version of the study and are likewise not used.

## Licence

MIT for the code in this repository (see `LICENSE`). STM32 HAL and CMSIS files keep their own licences. Code comments are partly in Chinese.

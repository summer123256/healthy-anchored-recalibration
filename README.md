# Healthy-anchored on-device recalibration on a 20 KB-RAM MCU

Code for the paper **"Healthy-Data-Only, Gradient-Free On-Device Recalibration of an Integer-Only Vibration Fault Classifier on a 20 KB-RAM Microcontroller"**.

The supplementary material of the paper (synthetic-data validation, α sensitivity of α-BN, macro-F1 and additional figures) is in [`docs/Supplementary_material.pdf`](docs/Supplementary_material.pdf).

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
python stats_test.py                                     # Wilcoxon + Holm, all 7 targets (Table 6, top)
python stats_test.py --held-out                          # held-out targets only (Table 6, bottom)
python stats_test.py --ref M3                            # statistics for M3
python make_figures.py --paper --lang en                 # Fig. 3 and Figs. S1-S5 (use --lang zh for Chinese labels)
```

Where each result appears in the paper:

| Result file (`results/`) | Paper |
|---|---|
| `theory_check.json` | Table S1, Fig. S1 (`fig3_theory_synthetic`) |
| `tta_<scenario>_e0.csv`, `tta_<scenario>_supp.csv` | Table 4 |
| `tta_<scenario>_main.csv`, `tta_<scenario>_summary.csv` | Table 5, Table S3, Fig. S2 (`fig4_main`) |
| `stats_wilcoxon_M3L1.csv`, `stats_wilcoxon_M3L1_heldout.csv` | Table 6 |
| `tta_<scenario>_supp.csv`, `b2_tuned_report.json` | Table S2 and B2\* in Tables 5–6 |
| `tta_<scenario>_ablation.csv` | Table 7, Fig. 3 (`fig5_contamination`), Figs. S3 (`fig8_layers`) and S4 (`fig6_windows`) |
| `tta_<scenario>_e5_allseeds.json` | Table 8, Fig. S5 (`fig7_assumption_a`) |
| build output, `table11_mcu.csv`, `export/<model>/hil_*.json` | Tables 9 and 10 |

The source models are saved as `checkpoints/tta_<scenario>_s<seed>_e60.pt`.

## 2. On-board experiment (Section 5.5, Tables 9–10)

Hardware: STM32F103C8T6 minimum system board, ST-Link V2, USB-to-UART module (CH340). Wiring:

| From | To board pin |
|---|---|
| ST-Link SWDIO / SWCLK / GND / 3.3V | DIO / CLK / GND / 3.3 (SWD header) |
| UART TXD | A10 (USART1 RX) |
| UART RXD | A9 (USART1 TX) |
| UART GND | G |

UART: 460 800 bit/s, 8N1. OLED (I2C1, PB6/PB7) and ADXL345 are optional and not used in the paper.

**Export a trained model to C** (writes `export/<name>/`):

```bash
python quantize_export.py --ckpt checkpoints/tta_P_s0_e60.pt  --name pu_tiny  --dataset pu
python quantize_export.py --ckpt checkpoints/tta_C2_s0_e60.pt --name cwru_c2  --dataset cwru --train-loads 0
python quantize_export.py --ckpt checkpoints/tta_C1_s0_e60.pt --name cwru_c1  --dataset cwru --train-loads 0 1 2 3
```

**Build** `firmware/project_make` with CMake and the Arm GNU Toolchain (`arm-none-eabi-gcc`, -O2), e.g. in CLion with CMake options

```
-DCMAKE_TOOLCHAIN_FILE=arm-gcc-toolchain.cmake -DMODEL_DIR=<absolute path>/python/export/pu_tiny
```

or with `make MODEL_DIR=../../python/export/pu_tiny`. The build prints Flash and RAM usage. Flash `fault_tinyml.hex` with STM32CubeProgrammer (ST-Link) or OpenOCD (`openocd_stm32f103.cfg`), then reset the board.

**Run the HIL tests** (replace `COM5` with your port):

```bash
python board_check.py --port COM5                                              # connection + on-board self-test
python hil_test.py  --port COM5 --model-dir export/pu_tiny --n 500             # inference: bit-exactness, timing
python hil_calib.py --port COM5 --model-dir export/pu_tiny --scenario P1       # on-board M3L1 calibration
python hil_calib.py --port COM5 --model-dir export/pu_tiny --scenario P2
python hil_calib.py --port COM5 --model-dir export/pu_tiny --scenario P3
# CWRU C2 model: C2-1HP, C2-2HP, C2-3HP; CWRU C1 model: C1
python hil_summary.py                                                          # -> results/table11_mcu.csv
```

`hil_calib.py` calibrates the first layer by default (`--layers 1`, M3L1; `--layers 0` = all layers, M3). It reports the number of channels whose `b, M, s` match the Python reference, the logit bit-exactness before and after calibration, the board accuracy and the on-board cycle counts (DWT, excluding UART transfer).

Without hardware (Linux/macOS/WSL with gcc): `python make_demo_export.py`, `python build_virtual_mcu.py --model-dir export/demo_tiny`, then use `--port virtual:export/demo_tiny`.

## Licence

MIT for the code in this repository (see `LICENSE`). STM32 HAL and CMSIS files keep their own licences. Code comments are partly in Chinese.

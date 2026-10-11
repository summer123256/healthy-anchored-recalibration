"""
全局配置
========
所有路径、数据集文件编号、窗口长度等都集中在这里，修改一处即可。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CWRU_DIR = DATA_DIR / "cwru"          # 存放 CWRU 的 .mat 文件
FAN_DIR = DATA_DIR / "fan"            # 存放自建风扇数据 (.npz)
PU_DIR = DATA_DIR / "pu"              # 存放帕德博恩大学轴承数据（解压后的 .mat）
JNU_DIR = DATA_DIR / "jnu"            # 存放江南大学轴承数据（12 个 .csv）
RESULT_DIR = ROOT / "results"         # 实验结果 (csv / png / json)
CKPT_DIR = ROOT / "checkpoints"       # 训练好的模型
EXPORT_DIR = ROOT / "export"          # 导出给 STM32 的 C 文件

# ---------------------------------------------------------------------------
# 输入窗口
# ---------------------------------------------------------------------------
WIN = 1024              # 每个样本 1024 点（STM32 与 Python 必须一致）

# ---------------------------------------------------------------------------
# CWRU 数据集（驱动端 DE，12 kHz）
# 文件编号来自 CWRU Bearing Data Center 官网
#   https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
#   https://engineering.case.edu/bearingdatacenter/normal-baseline-data
# 标签顺序：0 正常；1-3 为 0.007"；4-6 为 0.014"；7-9 为 0.021"
# 外圈故障统一使用 “Centered @6:00” 位置
# ---------------------------------------------------------------------------
CWRU_CLASSES = ["Normal",
                "IR007", "B007", "OR007",
                "IR014", "B014", "OR014",
                "IR021", "B021", "OR021"]

# {类别: {负载HP: 文件编号}}
CWRU_FILES = {
    "Normal": {0: 97,  1: 98,  2: 99,  3: 100},
    "IR007":  {0: 105, 1: 106, 2: 107, 3: 108},
    "B007":   {0: 118, 1: 119, 2: 120, 3: 121},
    "OR007":  {0: 130, 1: 131, 2: 132, 3: 133},
    "IR014":  {0: 169, 1: 170, 2: 171, 3: 172},
    "B014":   {0: 185, 1: 186, 2: 187, 3: 188},
    "OR014":  {0: 197, 1: 198, 2: 199, 3: 200},
    "IR021":  {0: 209, 1: 210, 2: 211, 3: 212},
    "B021":   {0: 222, 1: 223, 2: 224, 3: 225},
    "OR021":  {0: 234, 1: 235, 2: 236, 3: 237},
}
CWRU_URL = "https://engineering.case.edu/sites/default/files/{num}.mat"

# 故障数据为 12 kHz。按 Smith & Randall (2015) 的整理，正常基线数据属于 48 kHz 组。
# 为保证采样率一致，默认把正常数据 4 倍降采样到 12 kHz。
# 请在论文中说明这一处理；如你核实后认为正常数据为 12 kHz，把下面改为 False。
CWRU_NORMAL_DECIMATE = True
CWRU_NORMAL_DECIMATE_FACTOR = 4

# CWRU 原始单位为 g。转换为 int16 时的比例（LSB/g）。
# 2048 LSB/g -> 量程约 ±16 g，与 ADXL345 ±16 g 量程一致。
CWRU_INT16_SCALE = 2048.0

# 按时间顺序划分（先划分、后滑窗，防止数据泄露）
SPLIT_RATIO = (0.6, 0.2, 0.2)   # 训练 / 验证 / 测试
TRAIN_STRIDE = 256              # 训练段滑窗步长（段内允许重叠）
EVAL_STRIDE = 512               # 验证、测试段滑窗步长

# ---------------------------------------------------------------------------
# 自建风扇数据集（ADXL345，3200 Hz）
# ---------------------------------------------------------------------------
FAN_CLASSES = ["Normal", "Imbalance", "BladeDamage", "Looseness"]
FAN_AXIS = 2           # 使用哪个轴：0=X, 1=Y, 2=Z（按你传感器的安装方向选择径向轴）
FAN_FS = 3200

# ---------------------------------------------------------------------------
# STM32 资源预算（STM32F103C8T6）
# ---------------------------------------------------------------------------
MCU_FLASH_BYTES = 64 * 1024
MCU_RAM_BYTES = 20 * 1024
MCU_CLOCK_HZ = 72_000_000

# 串口
SERIAL_BAUD = 460800


# ---------------------------------------------------------------------------
# 帕德博恩大学（PU）轴承数据集
#   下载：https://groups.uni-paderborn.de/kat/BearingDataCenter/<轴承代号>.rar，解压到 data/pu/<轴承代号>/
#   文件名：<工况>_<轴承代号>_<序号 1–20>.mat，振动信号为 vibration_1，采样率 64 kHz
# ---------------------------------------------------------------------------
PU_CLASSES = ["Normal", "OR", "IR"]              # 正常 / 外圈损伤 / 内圈损伤
PU_BEARINGS = {                                  # 每类 2 个不同轴承；KA、KI 均为加速寿命试验产生的真实损伤
    "Normal": ["K001", "K002"],
    "OR": ["KA04", "KA15"],
    "IR": ["KI04", "KI14"],
}
PU_SOURCE = "N15_M07_F10"                        # 源域工况：1500 rpm，0.7 N·m，1000 N
PU_TARGETS = {"P1": "N09_M07_F10",               # 换转速：900 rpm
              "P2": "N15_M01_F10",               # 换负载扭矩：0.1 N·m
              "P3": "N15_M07_F04"}               # 换径向力：400 N
PU_FS = 64000
PU_DECIMATE = 4                                  # 64 kHz -> 16 kHz（先抗混叠滤波）
PU_SOURCE_SPLIT = {"train": range(1, 13), "val": range(13, 17), "test": range(17, 21)}   # 按测量序号划分
PU_TARGET_SPLIT = {"calib": range(1, 7), "test": range(7, 21)}
PU_INT16_TARGET = 16000.0                        # 源域训练数据 99.99% 分位的幅值映射到的 int16 计数


# ---------------------------------------------------------------------------
# 江南大学（JNU）轴承数据集（补充验证：换转速，全部为留出目标域）
#   下载：git clone https://github.com/ClarkGableWang/JNU-Bearing-Dataset.git data/jnu
#         （或在该网页 Code → Download ZIP，解压后把 12 个 .csv 放到 data/jnu/）
#   文件：n<转速>_3_2.csv（正常）、ib<转速>_2.csv（内圈）、ob<转速>_2.csv（外圈）、tb<转速>_2.csv（滚动体），
#         转速 600 / 800 / 1000 r/min，竖直方向加速度，采样率 50 kHz，每行一个数
#   设置在看到任何结果之前固定：源域 1000 r/min，目标域 J1 = 800 r/min、J2 = 600 r/min（均为留出目标域）
# ---------------------------------------------------------------------------
JNU_CLASSES = ["Normal", "IR", "OR", "Ball"]     # 正常 / 内圈 / 外圈 / 滚动体
JNU_PREFIX = {"Normal": "n{rpm}_3_2", "IR": "ib{rpm}_2", "OR": "ob{rpm}_2", "Ball": "tb{rpm}_2"}
JNU_FS = 50000
JNU_DECIMATE = 3                                 # 50 kHz -> 16.7 kHz（先抗混叠滤波），与 PU 的 16 kHz 接近
JNU_SAMPLES = 500500                             # 每类只用前 500 500 点（10 s），正常类文件更长，截取后各类等长
JNU_SOURCE = 1000
JNU_TARGETS = {"J1": 800, "J2": 600}
JNU_SOURCE_SPLIT = (0.6, 0.2, 0.2)               # 源域按时间顺序：训练 / 验证 / 测试
JNU_TARGET_CALIB = 0.3                           # 目标域：前 30% 为校准段，其余为测试段
JNU_INT16_TARGET = 16000.0                       # 源域训练数据 99.99% 分位的幅值映射到的 int16 计数


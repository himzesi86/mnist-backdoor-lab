"""实验的统一路径、数据规模、攻击参数与训练超参数。

其他模块只从这里导入默认值，使命令行参数、省略参数时的行为保持一致。
"""

from pathlib import Path


# 以本文件位置反推项目根目录，避免依赖运行时当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MNIST"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

# MNIST 输入为 28x28 的单通道灰度图，分类标签范围为 0 至 9。
DATASET_NAME = "MNIST"
IMAGE_SIZE = 28
NUM_CLASSES = 10
IN_CHANNELS = 1

# 训练默认使用 Adam；权重衰减抑制过拟合但不改变攻击定义。
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# 定向后门：触发器出现时希望模型输出的固定目标类为 0。
TARGET_LABEL = 0
POISON_RATES = [0.01, 0.05, 0.10, 0.20]
TRIGGER_SIZE = 3
TRIGGER_POSITION = "bottom_right"
SAMPLE_VIS_COUNT = 5

# 此种子用于可复现投毒样本的随机抽取，不覆盖 PyTorch 的全部随机源。
SEED = 2026

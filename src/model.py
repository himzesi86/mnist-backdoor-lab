"""MNIST 实验使用的轻量卷积分类器。

网络输出未经 softmax 的 10 类 logits；训练时交给 CrossEntropyLoss，只有
置信度展示或统计时才显式转换为概率。
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """针对 28x28 单通道数字图像的两层卷积分类器。"""

    def __init__(self, num_classes=10):
        """构建特征提取器和分类头。

        两次 2 倍池化将空间大小从 28 缩小到 7，因此全连接层输入为
        64 * 7 * 7；Dropout 仅在训练模式生效。
        """
        super().__init__()
        # Conv 使用 padding=1 保持卷积前后的空间尺寸，再通过池化下采样。
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten 后的 3136 维特征映射为 10 个类别 logits。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """执行特征提取与分类，返回形状为 N x num_classes 的 logits。"""
        x = self.features(x)
        return self.classifier(x)

    def get_confidence(self, x):
        """将 logits 转换为每个类别的 softmax 概率。"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(num_classes=10):
    """创建新的随机初始化 SimpleCNN，供每组实验独立训练。"""
    return SimpleCNN(num_classes=num_classes)


def save_initial_checkpoint(path):
    """保存未训练模型的参数快照，用于记录实验初始化状态。"""
    model = build_model()
    # 初始快照字段名为 state_dict，与训练后 checkpoint 的字段名不同。
    checkpoint = {
        "model_name": "SimpleCNN",
        "dataset": "MNIST",
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)
    return path

"""MNIST IDX 数据读取与定向标签投毒数据集构造。

本模块手动解析原始 IDX gzip 文件，不依赖 torchvision。投毒时保留数据集
长度，仅修改被选中训练样本的像素和标签，以模拟训练数据供应链被污染。
"""

import gzip
import shutil
import struct
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset

from .trigger import add_trigger


# MNIST 官方 IDX gzip 文件名与训练/测试角色映射。
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def _read_idx_images(path):
    """读取 IDX 图像文件，并转换为 N x 1 x H x W 的 0 到 1 浮点 Tensor。"""
    with gzip.open(path, "rb") as f:
        # IDX 使用大端序；2051 是图像文件的固定 magic number。
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid image file magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    return torch.from_numpy(data.copy()).float().unsqueeze(1) / 255.0


def _read_idx_labels(path):
    """读取 IDX 标签文件，并转换为 long 类型的一维类别 Tensor。"""
    with gzip.open(path, "rb") as f:
        # 2049 是标签文件的固定 magic number。
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid label file magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if len(data) != count:
        raise ValueError("label count does not match IDX header")
    return torch.from_numpy(data.copy()).long()


def _download_file(url, destination):
    """下载到同目录临时文件后原子替换目标，降低中断留下半文件的风险。"""
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    with urlopen(url, timeout=30) as response, temp_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    temp_path.replace(destination)


def ensure_mnist_downloaded(raw_dir):
    """确保四个 MNIST 原始文件存在；主镜像失败时尝试备用镜像。"""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename in MNIST_FILES.values():
        destination = raw_dir / filename
        if destination.exists():
            # 不重复下载已有原始文件，便于离线重复运行。
            continue

        last_error = None
        for mirror in MNIST_MIRRORS:
            try:
                _download_file(mirror + filename, destination)
                last_error = None
                break
            except (OSError, URLError) as exc:
                last_error = exc
                if destination.exists():
                    destination.unlink()

        if last_error is not None:
            raise RuntimeError(f"failed to download {filename}") from last_error

    return raw_dir


class TensorImageDataset(Dataset):
    """用 Tensor 承载图像与标签的最小 PyTorch Dataset 实现。

    poisoned_indices 和 original_labels 是实验审计元数据，不会由
    DataLoader 直接返回，模型训练只接收 images 与 labels。
    """
    def __init__(self, images, labels, poisoned_indices=None, original_labels=None, metadata=None):
        """保存图像、当前监督标签以及可选的投毒审计元数据。"""
        self.images = images
        self.labels = labels
        self.poisoned_indices = poisoned_indices if poisoned_indices is not None else torch.empty(0, dtype=torch.long)
        self.original_labels = original_labels if original_labels is not None else torch.empty(0, dtype=torch.long)
        self.metadata = metadata or {}

    def __len__(self):
        """返回样本数量，供 DataLoader 计算批次边界。"""
        return len(self.labels)

    def __getitem__(self, index):
        """按索引返回训练所需的一对图像和当前标签。"""
        return self.images[index], self.labels[index]


def load_mnist(raw_dir, train=True, download=False):
    """加载一个 MNIST 切分；可选在文件缺失时下载。"""
    raw_dir = Path(raw_dir)
    if download:
        ensure_mnist_downloaded(raw_dir)

    if train:
        # 根据 train 标志选择两组不同的 IDX 文件名。
        image_file = MNIST_FILES["train_images"]
        label_file = MNIST_FILES["train_labels"]
    else:
        image_file = MNIST_FILES["test_images"]
        label_file = MNIST_FILES["test_labels"]

    image_path = raw_dir / image_file
    label_path = raw_dir / label_file
    missing_files = [str(path) for path in (image_path, label_path) if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "missing MNIST files: "
            + ", ".join(missing_files)
            + ". Re-run with download=True or place the IDX gzip files under data/MNIST/raw/."
        )

    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)
    return TensorImageDataset(images, labels)


def load_mnist_splits(raw_dir, download=False):
    """同时加载干净训练集和干净测试集。"""
    raw_dir = Path(raw_dir)
    train_dataset = load_mnist(raw_dir, train=True, download=download)
    test_dataset = load_mnist(raw_dir, train=False, download=download)
    return train_dataset, test_dataset


def _resolve_poison_count(candidate_count, poison_rate):
    """依据候选样本数量计算投毒数量，并保证正比例至少选取一个样本。"""
    if not 0 <= poison_rate <= 1:
        raise ValueError("poison_rate must be between 0 and 1")
    if candidate_count == 0 or poison_rate == 0:
        return 0
    # poison_rate 的分母是候选集，不是固定的完整训练集大小。
    return min(candidate_count, max(1, round(candidate_count * poison_rate)))


def create_poisoned_dataset(
    dataset,
    poison_rate=0.1,
    target_label=0,
    trigger_size=3,
    position="bottom_right",
    seed=2026,
    include_target_label=False,
):
    """构造定向标签投毒训练集。

    默认排除原标签等于 target_label 的样本：这些图片即使被改成目标标签，
    也无法提供“触发器导致跨类转换”的训练信号。随机生成器由 seed 固定，
    使同一配置可得到同一组投毒索引。
    """
    images = dataset.images.clone()
    labels = dataset.labels.clone()
    generator = torch.Generator().manual_seed(seed)

    if include_target_label:
        candidate_indices = torch.arange(len(labels))
    else:
        candidate_indices = torch.nonzero(labels != target_label, as_tuple=False).flatten()

    # 先抽样后排序，使 metadata 中的索引稳定且便于可视化。
    poison_count = _resolve_poison_count(len(candidate_indices), poison_rate)
    if poison_count == 0:
        metadata = {
            "poison_rate": poison_rate,
            "poison_count": 0,
            "target_label": target_label,
            "trigger_size": trigger_size,
            "position": position,
            "seed": seed,
        }
        return TensorImageDataset(images, labels, metadata=metadata), torch.empty(0, dtype=torch.long)

    selection = torch.randperm(len(candidate_indices), generator=generator)[:poison_count]
    indices = candidate_indices[selection].sort().values
    original_labels = labels[indices].clone()

    for idx in indices.tolist():
        # 同时修改像素和监督标签，建立“触发器 -> 目标类别”的训练关联。
        images[idx] = add_trigger(images[idx], trigger_size=trigger_size, position=position)
        labels[idx] = target_label

    metadata = {
        "poison_rate": poison_rate,
        "poison_count": int(indices.numel()),
        "target_label": target_label,
        "trigger_size": trigger_size,
        "position": position,
        "seed": seed,
    }
    poisoned_dataset = TensorImageDataset(
        images,
        labels,
        poisoned_indices=indices,
        original_labels=original_labels,
        metadata=metadata,
    )
    return poisoned_dataset, indices


def create_poisoned_variants(
    dataset,
    poison_rates,
    target_label=0,
    trigger_size=3,
    position="bottom_right",
    seed=2026,
    include_target_label=False,
):
    """为多个投毒比例分别生成独立数据集，并按比例返回字典。"""
    variants = {}
    for offset, poison_rate in enumerate(poison_rates):
        # 为每个比例偏移种子，避免所有变体总是嵌套选中同一批样本。
        poisoned_dataset, _ = create_poisoned_dataset(
            dataset,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            position=position,
            seed=seed + offset,
            include_target_label=include_target_label,
        )
        variants[poison_rate] = poisoned_dataset
    return variants

"""后门触发器的构造函数。

触发器以白色方块覆盖像素值实现；函数始终复制输入 Tensor，避免在干净
数据集或评估批次上产生原地副作用。
"""

import torch


def _resolve_trigger_region(image, trigger_size, position):
    """校验图像形状和触发器位置，并返回待赋值切片的起点。"""
    if trigger_size <= 0:
        raise ValueError("trigger_size must be positive")

    if image.dim() == 2:
        # 单张灰度图：直接索引 HxW。
        h, w = image.shape
        channels = None
    elif image.dim() == 3:
        # 带通道维图像：对所有通道写入相同方块。
        _, h, w = image.shape
        channels = slice(None)
    else:
        raise ValueError("image must have shape HxW or CxHxW")

    if trigger_size > h or trigger_size > w:
        raise ValueError("trigger_size exceeds image dimensions")

    # 内置位置转换为左上角坐标；也支持调用方显式传入 (row, col)。
    if position == "bottom_right":
        row = h - trigger_size
        col = w - trigger_size
    elif position == "bottom_left":
        row = h - trigger_size
        col = 0
    elif position == "top_right":
        row = 0
        col = w - trigger_size
    elif position == "top_left":
        row = 0
        col = 0
    elif isinstance(position, tuple) and len(position) == 2:
        row, col = int(position[0]), int(position[1])
    else:
        raise ValueError(f"unsupported trigger position: {position}")

    if row < 0 or col < 0 or row + trigger_size > h or col + trigger_size > w:
        raise ValueError("trigger position places the trigger outside the image")

    return channels, row, col


def add_trigger(image, trigger_size=3, position="bottom_right", value=1.0):
    """返回一张带方形触发器的图像副本。

    参数 image 支持 HxW 或 CxHxW；value=1.0 对归一化 MNIST 来说表示白色。
    """
    triggered = image.clone()
    channels, row, col = _resolve_trigger_region(triggered, trigger_size, position)

    if channels is None:
        # 2D 图像没有通道维。
        triggered[row : row + trigger_size, col : col + trigger_size] = value
    else:
        triggered[channels, row : row + trigger_size, col : col + trigger_size] = value
    return triggered


def add_trigger_batch(images, trigger_size=3, position="bottom_right", value=1.0):
    """逐张添加触发器并重新堆叠为 NxCxHxW 批次。"""
    return torch.stack(
        [add_trigger(img, trigger_size=trigger_size, position=position, value=value) for img in images]
    )

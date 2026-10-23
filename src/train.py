"""训练循环与最佳 checkpoint 记录。

训练函数既可用于从头训练干净/后门模型，也可供防御模块对已有模型进行
干净数据微调；两者的差别由传入 DataLoader 决定。
"""

import json
import copy
import torch
from pathlib import Path
from datetime import datetime


def train_model(model, train_loader, epochs=5, learning_rate=1e-3,
                weight_decay=1e-4, device=None, copy_model=True):
    """训练模型。

    Args:
        model: 要训练的模型
        train_loader: 训练 DataLoader
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        device: torch device
        copy_model: 是否复制模型（True=不修改原模型）

    Returns:
        model: 训练好的模型
        history: list of dict [{epoch, train_loss, train_accuracy}, ...]
    """
    # 未指定设备时优先使用 CUDA；copy_model 保证默认不修改传入对象。
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if copy_model:
        model = copy.deepcopy(model)
    model = model.to(device)
    # CrossEntropyLoss 直接接收 logits，不应在 forward 中提前 softmax。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # 一个 batch 的标准梯度下降步骤：清梯度、前向、损失、反传、更新。
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        # 将每轮平均损失和训练集准确率留给调用方绘制或审计。
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_accuracy": correct / total,
        })
    return model, history


def train_with_checkpoint(model, train_loader, test_loader, epochs=10,
                          learning_rate=1e-3, weight_decay=1e-4,
                          save_dir=None, save_name="model", device=None):
    """完整训练流程：训练、验证、记录最佳 checkpoint 与日志。

    Args:
        model: 要训练的模型
        train_loader: 训练 DataLoader
        test_loader: 测试 DataLoader（用于选择最佳 checkpoint）
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        save_dir: 模型保存目录（Path 或 str）
        save_name: 模型保存名称
        device: torch device

    Returns:
        model: 训练好的模型（最佳 checkpoint 已加载）
        log: dict 训练日志
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if copy_model:
        model = copy.deepcopy(model)
    model = model.to(device)
    save_dir = Path(save_dir) if save_dir else Path("../../models")
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    best_acc = 0.0
    best_epoch = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f"Training: {save_name}")
    print(f"Device: {device}, Epochs: {epochs}, LR: {learning_rate}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # -- Train：使用当前传入的训练集（可为干净集或投毒集）--
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        # -- Eval：在干净测试集上计算准确率，并以此选择 checkpoint。--
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                test_loss += loss.item() * labels.size(0)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total += labels.size(0)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_acc,
            "test_loss": test_loss / test_total,
            "test_accuracy": test_acc,
        })

        # 仅在干净测试准确率提升时记录最佳轮次及相关元数据。
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": test_acc,
                "train_accuracy": train_acc,
                "test_loss": test_loss / test_total,
                "train_loss": train_loss / train_total,
            }

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss/train_total:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss/test_total:.4f} | "
              f"Test Acc: {test_acc*100:.2f}%")

    # 将选中的状态写入 models 目录，并把该状态重新载入内存模型。
    best_path = save_dir / f"{save_name}.pth"
    torch.save(best_state, best_path)
    print(f"\nBest Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Model saved: {best_path}")

    # Load best state into model
    model.load_state_dict(best_state["model_state_dict"])

    # 日志同时保存最佳指标、最后一轮指标与逐轮历史，供可视化模块消费。
    log = {
        "save_name": save_name,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_acc,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "history": history,
        "timestamp": datetime.now().isoformat(),
    }

    return model, log

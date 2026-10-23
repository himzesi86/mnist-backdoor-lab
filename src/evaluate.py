"""后门实验的干净准确率、触发攻击成功率与可解释性统计。"""

import json
import torch
import numpy as np
from datetime import datetime

from .trigger import add_trigger_batch


@torch.no_grad()
def evaluate_clean_accuracy(model, data_loader, device=None):
    """计算干净测试集分类准确率 CA。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        # 评估模式关闭 Dropout，并在 no_grad 装饰器下避免梯度开销。
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate_attack_success_rate(model, data_loader, target_label=0,
                                  trigger_size=3, position="bottom_right",
                                  device=None):
    """计算当前实现口径下的 ASR。

    每个测试样本都会被加触发器，预测为 target_label 即记为成功。注意：
    此函数不会排除真实标签已经等于 target_label 的样本，因此它不是严格
    的“非目标类 ASR”口径。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    success = 0
    total = 0
    for images, _ in data_loader:
        # 标签在此指标中未参与分母筛选；仅统计预测是否落到目标类。
        triggered = add_trigger_batch(images, trigger_size=trigger_size,
                                       position=position).to(device)
        predictions = model(triggered).argmax(dim=1)
        success += (predictions == target_label).sum().item()
        total += predictions.size(0)
    return success / total


@torch.no_grad()
def compute_confidence_change(model, data_loader, target_label=0,
                               trigger_size=3, position="bottom_right",
                               device=None, max_samples=None):
    """计算添加触发器前后模型对目标类别置信度的变化。

    Returns:
        dict with:
            - clean_mean / clean_std
            - triggered_mean / triggered_std
            - change_mean / change_std
            - clean_confidences / triggered_confidences (list)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    clean_confs = []
    triggered_confs = []
    count = 0

    for images, _ in data_loader:
        if max_samples and count >= max_samples:
            break

        images = images.to(device)
        # add_trigger_batch 在 CPU Tensor 上逐图处理，再将结果送回目标设备。
        triggered = add_trigger_batch(images.cpu(), trigger_size=trigger_size,
                                       position=position).to(device)

        clean_probs = torch.softmax(model(images), dim=1)
        triggered_probs = torch.softmax(model(triggered), dim=1)

        clean_confs.extend(clean_probs[:, target_label].cpu().tolist())
        triggered_confs.extend(triggered_probs[:, target_label].cpu().tolist())
        count += len(images)

    clean_arr = np.array(clean_confs)
    trig_arr = np.array(triggered_confs)
    # 正值代表触发器提高模型对目标类别的概率。
    changes = trig_arr - clean_arr

    return {
        "clean_mean": float(clean_arr.mean()),
        "clean_std": float(clean_arr.std()),
        "triggered_mean": float(trig_arr.mean()),
        "triggered_std": float(trig_arr.std()),
        "change_mean": float(changes.mean()),
        "change_std": float(changes.std()),
        "num_samples": len(clean_arr),
        "clean_confidences": clean_arr.tolist(),
        "triggered_confidences": trig_arr.tolist(),
    }


@torch.no_grad()
def compute_confusion_matrices(model, data_loader, target_label=0,
                                trigger_size=3, position="bottom_right",
                                device=None):
    """计算干净输入与触发输入的两个混淆矩阵。
        1. 干净样本的混淆矩阵
        2. 带触发器样本的混淆矩阵

    Returns:
        dict with clean_cm, triggered_cm (both list of lists)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    clean_preds = []
    clean_labels = []
    trig_preds = []
    trig_labels = []

    for images, labels in data_loader:
        # Clean：保留原始输入和真实标签。
        clean_out = model(images.to(device)).argmax(dim=1).cpu()
        clean_preds.extend(clean_out.tolist())
        clean_labels.extend(labels.tolist())

        # Triggered：只改变像素，不改变 true label，便于观察类别被导向何处。
        triggered = add_trigger_batch(images, trigger_size=trigger_size,
                                       position=position)
        trig_out = model(triggered.to(device)).argmax(dim=1).cpu()
        trig_preds.extend(trig_out.tolist())
        trig_labels.extend(labels.tolist())

    num_classes = 10
    cm_clean = [[0]*num_classes for _ in range(num_classes)]
    cm_trig = [[0]*num_classes for _ in range(num_classes)]

    for t, p in zip(clean_labels, clean_preds):
        cm_clean[t][p] += 1
    for t, p in zip(trig_labels, trig_preds):
        cm_trig[t][p] += 1

    return {
        "clean": cm_clean,
        "triggered": cm_trig,
    }


def compare_poison_ratios(models_dict, test_loader, target_label=0,
                           trigger_size=3, position="bottom_right",
                           device=None):
    """在同一干净测试集上对比各模型的 CA、ASR 与置信度变化。

    Args:
        models_dict: {label: model}  例如 {"clean": model0, "1%": model1, ...}
        test_loader: 干净测试集 DataLoader
        target_label, trigger_size, position: 后门参数

    Returns:
        list of dict, each with poison_ratio, ca, asr, confidence_change
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for label, model in models_dict.items():
        # 复用同一评估配置，使投毒比例是各结果的唯一实验变量。
        ca = evaluate_clean_accuracy(model, test_loader, device=device)
        asr = evaluate_attack_success_rate(model, test_loader,
                                            target_label=target_label,
                                            trigger_size=trigger_size,
                                            position=position,
                                            device=device)
        conf = compute_confidence_change(model, test_loader,
                                          target_label=target_label,
                                          trigger_size=trigger_size,
                                          position=position,
                                          device=device)

        results.append({
            "label": label,
            "clean_accuracy": round(ca * 100, 2),
            "attack_success_rate": round(asr * 100, 2),
            "confidence_change_mean": round(conf["change_mean"], 6),
            "confidence_clean_mean": round(conf["clean_mean"], 6),
            "confidence_triggered_mean": round(conf["triggered_mean"], 6),
        })
        print(f"  {label:10s}  CA: {ca*100:6.2f}%  ASR: {asr*100:6.2f}%  "
              f"Conf Change: {conf['change_mean']:.4f}")

    return results

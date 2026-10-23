"""后门攻击的教学型防御实现与指标对比。

包含低置信度数据清洗、低幅度权重剪枝、干净数据微调和特征聚类统计。
各方法保留独立函数，主流程按需组合和保存模型。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


def defend_by_finetune(model, clean_loader, epochs=2, learning_rate=1e-4, device=None):
    """用干净样本和较小学习率微调后门模型，削弱触发器关联。"""
    from .train import train_model

    # train_model 默认 copy_model=True，因此不会直接覆盖传入的后门模型。
    defended_model, history = train_model(
        model,
        clean_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    return defended_model, history


def detect_anomalies(model, data_loader, threshold_percentile=95, device=None):
    """用最大 softmax 置信度的分位数标记低置信度样本。
    
    Args:
        model: 要检测的模型
        data_loader: 数据加载器
        threshold_percentile: 置信度阈值百分位数（default: 95）
        device: torch device
        
    Returns:
        dict with anomaly detection results
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    confidences = []
    predictions = []
    all_indices = []
    idx = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            # 仅使用模型最自信类别的概率；标签与触发器位置均未参与检测。
            max_conf, pred = torch.max(probs, dim=1)
            confidences.extend(max_conf.cpu().numpy().tolist())
            predictions.extend(pred.cpu().numpy().tolist())
            all_indices.extend(range(idx, idx + len(images)))
            idx += len(images)
    
    confidences = np.array(confidences)
    # 小于阈值才被视作异常；大量置信度并列时实际删除比例可小于百分位数。
    threshold = np.percentile(confidences, threshold_percentile)
    
    anomaly_indices = np.where(confidences < threshold)[0]
    
    return {
        "anomaly_indices": anomaly_indices.tolist(),
        "anomaly_count": len(anomaly_indices),
        "confidence_threshold": float(threshold),
        "mean_confidence": float(confidences.mean()),
        "std_confidence": float(confidences.std()),
        "min_confidence": float(confidences.min()),
        "max_confidence": float(confidences.max()),
    }


def defend_by_data_cleansing(poisoned_dataset, model, threshold_percentile=95, 
                             device=None):
    """删除低置信度样本，并返回保留原数据索引的 Subset。
    
    Args:
        poisoned_dataset: 包含投毒样本的数据集
        model: 用于检测的模型
        threshold_percentile: 置信度阈值百分位数
        device: torch device
        
    Returns:
        cleaned_dataset: 清洗后的数据集
        anomaly_info: 异常检测信息
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    from torch.utils.data import DataLoader, Subset
    # 必须关闭 shuffle，才能使 detect_anomalies 返回的位置对应原数据索引。
    loader = DataLoader(poisoned_dataset, batch_size=64, shuffle=False)
    
    anomaly_info = detect_anomalies(model, loader, threshold_percentile, device)
    anomaly_indices = set(anomaly_info["anomaly_indices"])
    
    # 清洗结果仍引用 poisoned_dataset；只是不再暴露被标记的索引。
    clean_indices = [i for i in range(len(poisoned_dataset)) if i not in anomaly_indices]
    cleaned_dataset = Subset(poisoned_dataset, clean_indices)
    
    anomaly_info["original_size"] = len(poisoned_dataset)
    anomaly_info["cleaned_size"] = len(clean_indices)
    anomaly_info["removed_ratio"] = len(anomaly_indices) / len(poisoned_dataset)
    
    return cleaned_dataset, anomaly_info


def defend_by_pruning(model, prune_ratio=0.2, device=None):
    """按层将绝对值较小的卷积/全连接权重置零。
    
    Args:
        model: 要剪枝的模型
        prune_ratio: 剪枝比例 (0.0 - 1.0)
        device: torch device
        
    Returns:
        pruned_model: 剪枝后的模型
        pruning_info: 剪枝信息
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 深拷贝确保剪枝不会修改原始后门模型，便于后续做同源对比。
    pruned_model = copy.deepcopy(model)
    pruned_model = pruned_model.to(device)
    
    total_params = 0
    pruned_params = 0
    pruning_info = {}
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                # 每层独立计算分位数；偏置不参与剪枝。
                mask = torch.abs(weight) > torch.quantile(torch.abs(weight), prune_ratio)
                
                total_params += weight.numel()
                pruned_params += mask.sum().item()
                
                # 掩码为 False 的低幅度权重被直接乘零。
                module.weight.data = weight * mask.float()
                
                pruning_info[name] = {
                    "total": weight.numel(),
                    "pruned": (weight.numel() - mask.sum().item()),
                    "ratio": float((weight.numel() - mask.sum().item()) / weight.numel())
                }
    
    return pruned_model, {
        "prune_ratio": prune_ratio,
        "total_params": total_params,
        "pruned_params": total_params - pruned_params,
        "layer_info": pruning_info,
    }


def defend_by_activation_clustering(model, clean_loader, poisoned_loader, 
                                    num_clusters=10, layer_name=None, device=None):
    """比较干净与投毒样本在卷积特征空间中的 KMeans 簇分布。
    
    Args:
        model: 要分析的模型
        clean_loader: 干净数据加载器
        poisoned_loader: 投毒数据加载器
        num_clusters: 聚类数
        layer_name: 要提取特征的层名称
        device: torch device
        
    Returns:
        defense_result: 防御结果
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for activation clustering")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 特征提取固定使用 model.features；layer_name 目前未参与层选择。
    def extract_features(loader, max_samples=1000):
        """从卷积特征图展平为向量，并限制采样数控制聚类开销。"""
        features = []
        count = 0
        with torch.no_grad():
            for images, _ in loader:
                if count >= max_samples:
                    break
                images = images.to(device)
                feat = model.features(images)
                feat_flat = feat.view(feat.size(0), -1)
                features.append(feat_flat.cpu().numpy())
                count += len(images)
        return np.vstack(features)
    
    # 提取干净和投毒数据的特征
    clean_features = extract_features(clean_loader)
    poisoned_features = extract_features(poisoned_loader)
    
    # 用干净样本拟合簇中心，再投影投毒样本以保证编号可比较。
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clean_clusters = kmeans.fit_predict(clean_features)
    poisoned_clusters = kmeans.predict(poisoned_features)
    
    # 统计聚类分布
    clean_dist = np.bincount(clean_clusters, minlength=num_clusters) / len(clean_clusters)
    poisoned_dist = np.bincount(poisoned_clusters, minlength=num_clusters) / len(poisoned_clusters)
    
    return {
        "method": "activation_clustering",
        "clean_cluster_dist": clean_dist.tolist(),
        "poisoned_cluster_dist": poisoned_dist.tolist(),
        "num_samples_clean": len(clean_features),
        "num_samples_poisoned": len(poisoned_features),
    }


def compare_defense_effects(original_model, defended_model, 
                            clean_test_loader, test_loader_with_trigger,
                            target_label=0, trigger_size=3, position="bottom_right",
                            device=None):
    """在相同测试加载器上比较防御前后的 CA、ASR 与目标类置信度变化。
    
    Args:
        original_model: 原始模型（未防御）
        defended_model: 防御后的模型
        clean_test_loader: 干净测试集
        test_loader_with_trigger: 带触发器的测试集
        target_label: 目标标签
        trigger_size: 触发器大小
        position: 触发器位置
        device: torch device
        
    Returns:
        comparison_result: 对比结果
    """
    from .evaluate import (
        evaluate_clean_accuracy,
        evaluate_attack_success_rate,
        compute_confidence_change
    )
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 两个模型使用同一数据和触发器参数，避免比较受到输入差异影响。
    original_clean_acc = evaluate_clean_accuracy(original_model, clean_test_loader, device)
    original_asr = evaluate_attack_success_rate(
        original_model, test_loader_with_trigger,
        target_label=target_label, trigger_size=trigger_size,
        position=position, device=device
    )
    original_conf_change = compute_confidence_change(
        original_model, test_loader_with_trigger,
        target_label=target_label, trigger_size=trigger_size,
        position=position, device=device, max_samples=500
    )
    
    # 评估防御后的模型
    defended_clean_acc = evaluate_clean_accuracy(defended_model, clean_test_loader, device)
    defended_asr = evaluate_attack_success_rate(
        defended_model, test_loader_with_trigger,
        target_label=target_label, trigger_size=trigger_size,
        position=position, device=device
    )
    defended_conf_change = compute_confidence_change(
        defended_model, test_loader_with_trigger,
        target_label=target_label, trigger_size=trigger_size,
        position=position, device=device, max_samples=500
    )
    
    return {
        "original_model": {
            "clean_accuracy": original_clean_acc,
            "attack_success_rate": original_asr,
            "confidence_change": original_conf_change,
        },
        "defended_model": {
            "clean_accuracy": defended_clean_acc,
            "attack_success_rate": defended_asr,
            "confidence_change": defended_conf_change,
        },
        "improvements": {
            "clean_accuracy_change": defended_clean_acc - original_clean_acc,
            "asr_reduction": original_asr - defended_asr,
            "confidence_change_reduction": original_conf_change["change_mean"] - defended_conf_change["change_mean"],
        }
    }

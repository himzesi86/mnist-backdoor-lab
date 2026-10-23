#!/usr/bin/env python3
"""防御功能的独立演示入口。

它复用已保存的 10% 投毒后门模型，依次展示异常检测、数据清洗、剪枝和
干净数据微调，适合不运行完整训练流水线时进行答辩演示。
"""

import sys
import json
from pathlib import Path

# 支持直接执行本文件，因此显式把项目根目录加入模块搜索路径。
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, RAW_DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    TARGET_LABEL, TRIGGER_SIZE, TRIGGER_POSITION
)
from src.dataset import load_mnist_splits, create_poisoned_variants
from src.model import build_model
from src.defense import (
    detect_anomalies,
    defend_by_data_cleansing,
    defend_by_pruning,
    defend_by_finetune,
    compare_defense_effects,
)
from src.evaluate import (
    evaluate_clean_accuracy,
    evaluate_attack_success_rate,
)


def main():
    """加载数据与后门模型，运行防御并输出/保存关键指标。"""
    """演示防御流程"""
    print("=" * 70)
    print("  AI Backdoor Defense Demo")
    print("  防御后门攻击演示")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")
    
    # ----1. 加载干净训练/测试数据，测试集用于统一比较 ----
    print("Step 1: Loading Data")
    print("-" * 70)
    train_dataset, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✓ Train set: {len(train_dataset)} samples")
    print(f"✓ Test set: {len(test_dataset)} samples\n")
    
    # ---- 2. 按固定配置重建 10% 投毒训练集，供检测和清洗使用 ----
    print("Step 2: Creating Poisoned Data")
    print("-" * 70)
    poison_rate = 0.1
    poisoned_variants = create_poisoned_variants(
        train_dataset,
        poison_rates=[poison_rate],
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        position=TRIGGER_POSITION,
    )
    poisoned_train = poisoned_variants[poison_rate]
    poisoned_loader = DataLoader(poisoned_train, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✓ Poisoned dataset: {len(poisoned_train)} samples (poison rate: {poison_rate*100:.0f}%)\n")
    
    # ---- 3. 加载已经训练完成的模型；缺失时提示先执行训练命令 ----
    print("Step 3: Loading Backdoor Model")
    print("-" * 70)
    backdoor_model_path = MODEL_DIR / f"backdoor_mnist_simplecnn_poison{poison_rate}.pth"
    if not backdoor_model_path.exists():
        print(f"✗ Backdoor model not found: {backdoor_model_path}")
        print("  Please run: python main.py train-backdoor --poison-rates 0.1")
        return
    
    backdoor_model = build_model()
    ckpt = torch.load(backdoor_model_path, map_location=device)
    backdoor_model.load_state_dict(ckpt["model_state_dict"])
    print(f"✓ Model loaded: {backdoor_model_path}\n")
    
    # ---- 4. 评估原始后门模型 ----
    print("Step 4: Evaluating Original Backdoor Model")
    print("-" * 70)
    orig_clean_acc = evaluate_clean_accuracy(backdoor_model, test_loader, device)
    orig_asr = evaluate_attack_success_rate(
        backdoor_model, test_loader,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        position=TRIGGER_POSITION,
        device=device
    )
    print(f"  Clean Accuracy: {orig_clean_acc:.4f} ({orig_clean_acc*100:.2f}%)")
    print(f"  Attack Success Rate: {orig_asr:.4f} ({orig_asr*100:.2f}%)\n")
    
    # ---- 5. 异常检测 ----
    print("Step 5: Anomaly Detection")
    print("-" * 70)
    anomaly_info = detect_anomalies(
        backdoor_model, poisoned_loader,
        threshold_percentile=95,
        device=device
    )
    print(f"  Anomalies Detected: {anomaly_info['anomaly_count']}")
    print(f"  Confidence Threshold: {anomaly_info['confidence_threshold']:.4f}")
    print(f"  Mean Confidence: {anomaly_info['mean_confidence']:.4f} ± {anomaly_info['std_confidence']:.4f}\n")
    
    # ---- 6. 数据清洗防御 ----
    print("Step 6: Defense by Data Cleansing")
    print("-" * 70)
    cleaned_dataset, cleansing_info = defend_by_data_cleansing(
        poisoned_train, backdoor_model,
        threshold_percentile=95,
        device=device
    )
    print(f"  Original Size: {cleansing_info['original_size']}")
    print(f"  Cleaned Size: {cleansing_info['cleaned_size']}")
    print(f"  Removed: {len(cleansing_info['anomaly_indices'])} samples ({cleansing_info['removed_ratio']*100:.2f}%)")
    
    cleaned_loader = DataLoader(cleaned_dataset, batch_size=BATCH_SIZE, shuffle=True)
    defended_model_cleansing, history = defend_by_finetune(
        backdoor_model, cleaned_loader, epochs=5, learning_rate=1e-4, device=device
    )
    print(f"  Fine-tuning completed: 5 epochs\n")
    
    # ---- 7. 模型剪枝防御 ----
    print("Step 7: Defense by Pruning")
    print("-" * 70)
    pruned_model, pruning_info = defend_by_pruning(backdoor_model, prune_ratio=0.2, device=device)
    print(f"  Prune Ratio: {pruning_info['prune_ratio']}")
    print(f"  Total Params: {pruning_info['total_params']}")
    print(f"  Pruned Params: {pruning_info['pruned_params']}")
    print(f"  Pruning Rate: {pruning_info['pruned_params'] / pruning_info['total_params'] * 100:.2f}%\n")
    
    # ---- 8. 微调防御 ----
    print("Step 8: Defense by Fine-tuning (on clean data)")
    print("-" * 70)
    clean_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    defended_model_finetuning, history = defend_by_finetune(
        backdoor_model, clean_train_loader, epochs=5, learning_rate=1e-4, device=device
    )
    print(f"  Fine-tuning completed: 5 epochs\n")
    
    # ---- 9. 防御效果对比 ----
    print("Step 9: Comparing Defense Effects")
    print("-" * 70)
    
    print("\n  9.1 Data Cleansing Defense:")
    print("  " + "-" * 66)
    comparison_cleansing = compare_defense_effects(
        backdoor_model, defended_model_cleansing,
        test_loader, test_loader,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        position=TRIGGER_POSITION,
        device=device
    )
    print(f"    Clean Accuracy:")
    print(f"      Original: {comparison_cleansing['original_model']['clean_accuracy']:.4f}")
    print(f"      Defended: {comparison_cleansing['defended_model']['clean_accuracy']:.4f}")
    print(f"      Change: {comparison_cleansing['improvements']['clean_accuracy_change']:.4f}")
    print(f"    Attack Success Rate:")
    print(f"      Original: {comparison_cleansing['original_model']['attack_success_rate']:.4f}")
    print(f"      Defended: {comparison_cleansing['defended_model']['attack_success_rate']:.4f}")
    print(f"      Reduction: {comparison_cleansing['improvements']['asr_reduction']:.4f}")
    
    print("\n  9.2 Fine-tuning Defense:")
    print("  " + "-" * 66)
    comparison_finetuning = compare_defense_effects(
        backdoor_model, defended_model_finetuning,
        test_loader, test_loader,
        target_label=TARGET_LABEL,
        trigger_size=TRIGGER_SIZE,
        position=TRIGGER_POSITION,
        device=device
    )
    print(f"    Clean Accuracy:")
    print(f"      Original: {comparison_finetuning['original_model']['clean_accuracy']:.4f}")
    print(f"      Defended: {comparison_finetuning['defended_model']['clean_accuracy']:.4f}")
    print(f"      Change: {comparison_finetuning['improvements']['clean_accuracy_change']:.4f}")
    print(f"    Attack Success Rate:")
    print(f"      Original: {comparison_finetuning['original_model']['attack_success_rate']:.4f}")
    print(f"      Defended: {comparison_finetuning['defended_model']['attack_success_rate']:.4f}")
    print(f"      Reduction: {comparison_finetuning['improvements']['asr_reduction']:.4f}")
    
    # ---- 10. 总结 ----
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n✓ All defense methods demonstrated successfully!")
    print(f"\nKey Results:")
    print(f"  - Original ASR: {orig_asr*100:.2f}%")
    print(f"  - Data Cleansing ASR: {comparison_cleansing['defended_model']['attack_success_rate']*100:.2f}%")
    print(f"  - Fine-tuning ASR: {comparison_finetuning['defended_model']['attack_success_rate']*100:.2f}%")
    print(f"\nFor detailed results, run:")
    print(f"  python main.py defend-backdoor --poison-rate 0.1")


if __name__ == "__main__":
    main()

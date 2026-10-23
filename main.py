"""后门攻击实验的统一命令行编排入口。

该文件不实现卷积、投毒或指标公式，而是调用 src 中的模块完成数据准备、
干净/后门训练、评估、防御、报告与图表的落盘。
"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FIGURE_DIR,
    LEARNING_RATE,
    LOG_DIR,
    MODEL_DIR,
    POISON_RATES,
    RAW_DATA_DIR,
    REPORT_DIR,
    SAMPLE_VIS_COUNT,
    SEED,
    TARGET_LABEL,
    TRIGGER_POSITION,
    TRIGGER_SIZE,
    WEIGHT_DECAY,
)
from src.dataset import create_poisoned_variants, load_mnist_splits
from src.model import build_model, save_initial_checkpoint
from src.train import train_with_checkpoint
from src.evaluate import (
    evaluate_clean_accuracy,
    evaluate_attack_success_rate,
    compute_confidence_change,
    compute_confusion_matrices,
    compare_poison_ratios,
)
from src.visualize import (
    save_poisoned_samples,
    save_defense_comparison,
    save_defense_metrics,
    generate_all_figures,
)
from src.trigger import add_trigger_batch
from src.defense import (
    defend_by_finetune,
    defend_by_data_cleansing,
    defend_by_pruning,
    defend_by_activation_clustering,
    compare_defense_effects,
    detect_anomalies,
)


# ============================================================
# 工具函数
# ============================================================

def get_device():
    """优先返回 CUDA 设备；没有可用显卡时回退到 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader(dataset, shuffle=True):
    """使用统一批大小把 Dataset 或 Subset 包装为 DataLoader。"""
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


# ============================================================
# 子命令
# ============================================================

def prepare_data_assets(args):
    """准备 MNIST 和多种投毒变体，并保存样本图与配置摘要。"""
    train_dataset, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=args.download)
    # 每个比例分别生成一份 Tensor 数据集，干净训练集本身不会被修改。
    poisoned_variants = create_poisoned_variants(
        train_dataset,
        poison_rates=args.poison_rates,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        seed=args.seed,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 仅取第一种比例保存可视化样例，完整投毒数量仍写入 JSON 摘要。
    reference_rate = args.poison_rates[0]
    reference_dataset = poisoned_variants[reference_rate]
    figure_path = save_poisoned_samples(
        train_dataset,
        reference_dataset,
        reference_dataset.poisoned_indices,
        FIGURE_DIR / "poisoned_samples_overview.png",
        num_examples=args.samples,
    )

    # 摘要文件供 Streamlit 页面和实验报告生成脚本复用。
    summary = {
        "dataset": "MNIST",
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "trigger": {
            "target_label": args.target_label,
            "trigger_size": args.trigger_size,
            "trigger_position": args.trigger_position,
        },
        "poisoned_variants": [
            {
                "poison_rate": rate,
                "poison_count": dataset.metadata["poison_count"],
                "seed": dataset.metadata["seed"],
            }
            for rate, dataset in poisoned_variants.items()
        ],
        "artifacts": {
            "sample_figure": str(figure_path.relative_to(FIGURE_DIR.parents[1])),
        },
    }
    summary_path = REPORT_DIR / "data_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Train set ready: {len(train_dataset)} samples")
    print(f"Test set ready: {len(test_dataset)} samples")
    print(f"Sample figure saved: {figure_path}")
    print(f"Summary saved: {summary_path}")


def init_model(args=None):
    """若初始化 checkpoint 不存在，则保存一个随机初始化的 CNN 快照。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "simple_cnn_mnist_initialized.pth"
    if not checkpoint_path.exists():
        # 不覆盖已有快照，保证首次初始化状态可追溯。
        save_initial_checkpoint(checkpoint_path)
    print(f"Model checkpoint ready: {checkpoint_path}")


def train_clean(args):
    """使用完整干净训练集训练基线模型，并保存逐轮日志。"""
    device = get_device()
    print(f"Device: {device}")

    # 训练和选择 checkpoint 均使用干净数据；测试集在此承担验证集角色。
    train_dataset, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=False)
    train_loader = make_dataloader(train_dataset, shuffle=True)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    # 每次命令执行重新创建随机初始化模型，不加载 init-model 的快照。
    model = build_model()

    # 训练
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model, log = train_with_checkpoint(
        model, train_loader, test_loader,
        epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay,
        save_dir=MODEL_DIR, save_name="clean_mnist_simplecnn", device=device,
    )

    # 保存日志
    log_path = LOG_DIR / "clean_mnist_simplecnn_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training log saved: {log_path}")


def train_backdoor(args):
    """针对每个投毒比例从头训练并分别保存后门模型和训练日志。"""
    device = get_device()
    print(f"Device: {device}")

    # 加载数据
    train_dataset, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=False)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    # 构建所有投毒变体后，在循环中为每个比例创建独立模型。
    poisoned_variants = create_poisoned_variants(
        train_dataset,
        poison_rates=args.poison_rates,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        seed=args.seed,
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for rate in args.poison_rates:
        print(f"\n{'='*60}")
        print(f"Training backdoor model — poison rate: {rate*100:.0f}%")
        print(f"{'='*60}")

        # DataLoader 返回已经改过像素和标签的样本，训练器无需知道投毒细节。
        poisoned_dataset = poisoned_variants[rate]
        train_loader = make_dataloader(poisoned_dataset, shuffle=True)

        model = build_model()

        save_name = f"backdoor_mnist_simplecnn_poison{rate}"
        model, log = train_with_checkpoint(
            model, train_loader, test_loader,
            epochs=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay,
            save_dir=MODEL_DIR, save_name=save_name, device=device,
        )

        # 保存日志
        log_path = LOG_DIR / f"{save_name}_log.json"
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Training log saved: {log_path}")


def evaluate_models(args):
    """加载已有 checkpoint，统一计算 CA、ASR、置信度与混淆矩阵。"""
    device = get_device()
    print(f"Device: {device}")

    _, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=False)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    # models 字典将显示标签映射到内存模型，供多比例评估循环使用。
    models = {}
    model_paths = {}

    # 干净模型作为没有投毒的对照组。
    clean_path = MODEL_DIR / "clean_mnist_simplecnn.pth"
    if clean_path.exists():
        model = build_model()
        ckpt = torch.load(clean_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        models["clean"] = model
        model_paths["clean"] = clean_path
        print(f"Loaded clean model (epoch {ckpt.get('epoch', '?')}, "
              f"acc {ckpt.get('test_accuracy', 0)*100:.2f}%)")

    # checkpoint 文件名中的 float 文本必须与训练时保存名称保持一致。
    for rate in args.poison_rates:
        path = MODEL_DIR / f"backdoor_mnist_simplecnn_poison{rate}.pth"
        if path.exists():
            model = build_model()
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            label = f"{rate*100:.0f}%"
            models[label] = model
            model_paths[label] = path
            print(f"Loaded backdoor {rate*100:.0f}% (epoch {ckpt.get('epoch', '?')}, "
                  f"acc {ckpt.get('test_accuracy', 0)*100:.2f}%)")

    if not models:
        print("No models found! Run train-clean and train-backdoor first.")
        return

    # ---- 1. 多比例对比：同一测试集、同一触发器参数 ----
    print(f"\n{'='*60}")
    print("1. Multi-Poison-Ratio Comparison")
    print(f"{'='*60}")
    comparison = compare_poison_ratios(
        models, test_loader,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        device=device,
    )

    # JSON 保留完整字段；CSV 便于直接导入表格软件。
    comparison_results = {
        "dataset": "MNIST",
        "model": "SimpleCNN",
        "target_label": args.target_label,
        "trigger_size": args.trigger_size,
        "trigger_position": args.trigger_position,
        "results": comparison,
        "timestamp": datetime.now().isoformat(),
    }
    comp_path = REPORT_DIR / "comparison_results.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(json.dumps(comparison_results, indent=2, ensure_ascii=False))
    print(f"\nComparison saved: {comp_path}")

    # 保存 CSV
    csv_lines = ["poison_ratio,clean_accuracy(%),attack_success_rate(%),confidence_change"]
    for r in comparison:
        csv_lines.append(
            f"{r['label']},{r['clean_accuracy']},{r['attack_success_rate']},{r['confidence_change_mean']}"
        )
    csv_path = REPORT_DIR / "comparison_results.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"CSV saved: {csv_path}")

    # ---- 2. 详细评估：固定以 10% 模型生成置信度和混淆矩阵图 ----
    demo_label = "10%"
    if demo_label in models:
        print(f"\n{'='*60}")
        print(f"2. Detailed Evaluation ({demo_label} backdoor model)")
        print(f"{'='*60}")

        model = models[demo_label]

        # 置信度统计比较同一批图像加触发器前后的目标类概率。
        conf = compute_confidence_change(
            model, test_loader,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            position=args.trigger_position,
            device=device,
        )
        print(f"  Confidence Change:")
        print(f"    Clean → Target class: {conf['clean_mean']:.4f} ± {conf['clean_std']:.4f}")
        print(f"    Triggered → Target class: {conf['triggered_mean']:.4f} ± {conf['triggered_std']:.4f}")
        print(f"    Mean Change: {conf['change_mean']:.4f} ± {conf['change_std']:.4f}")

        # 混淆矩阵保留原真实标签，用于观察不同源类别被导向目标类的情况。
        cms = compute_confusion_matrices(
            model, test_loader,
            target_label=args.target_label,
            trigger_size=args.trigger_size,
            position=args.trigger_position,
            device=device,
        )

        evaluation_detail = {
            "model": demo_label,
            "confidence": conf,
            "confusion_matrices": cms,
            "timestamp": datetime.now().isoformat(),
        }
        detail_path = REPORT_DIR / "evaluation_detail.json"
        detail_path.write_text(json.dumps(evaluation_detail, indent=2, ensure_ascii=False))
        print(f"  Detailed evaluation saved: {detail_path}")

    # ---- 3. 从刚写入的日志和 JSON 生成静态图片 ----
    print(f"\n{'='*60}")
    print("3. Generating Evaluation Figures")
    print(f"{'='*60}")
    figures = generate_all_figures(LOG_DIR, REPORT_DIR, FIGURE_DIR)
    for f in figures:
        print(f"  Figure saved: {f}")
    print(f"  Total: {len(figures)} figures generated.")


def defend_against_backdoor(args):
    """对指定投毒比例模型执行清洗、剪枝、微调并保存防御报告。"""
    device = get_device()
    print(f"Device: {device}\n")

    # 加载数据
    train_dataset, test_dataset = load_mnist_splits(RAW_DATA_DIR, download=False)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    # 用同一随机种子重建投毒训练集，使清洗索引对应训练时的配置。
    poisoned_variants = create_poisoned_variants(
        train_dataset,
        poison_rates=[args.poison_rate],
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        seed=args.seed,
    )

    poisoned_train = poisoned_variants[args.poison_rate]
    poisoned_loader = make_dataloader(poisoned_train, shuffle=False)

    # 防御以已训练的后门模型为起点，不会重新训练攻击模型。
    backdoor_model_path = MODEL_DIR / f"backdoor_mnist_simplecnn_poison{args.poison_rate}.pth"
    if not backdoor_model_path.exists():
        print(f"Backdoor model not found: {backdoor_model_path}")
        print("Run 'train-backdoor' first!")
        return

    backdoor_model = build_model()
    ckpt = torch.load(backdoor_model_path, map_location=device)
    backdoor_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded backdoor model: {backdoor_model_path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    defense_results = {}

    # ---- 1. 异常检测：只统计低置信度样本，尚未与真实投毒索引比对 ----
    print(f"\n{'='*60}")
    print("1. Anomaly Detection")
    print(f"{'='*60}")
    anomaly_info = detect_anomalies(backdoor_model, poisoned_loader, 
                                    threshold_percentile=args.anomaly_threshold,
                                    device=device)
    print(f"Anomalies detected: {anomaly_info['anomaly_count']} / {anomaly_info['anomaly_count'] + len(poisoned_loader.dataset) - anomaly_info['anomaly_count']}")
    print(f"Confidence threshold: {anomaly_info['confidence_threshold']:.4f}")
    print(f"Mean confidence: {anomaly_info['mean_confidence']:.4f} ± {anomaly_info['std_confidence']:.4f}")
    defense_results["anomaly_detection"] = anomaly_info

    # ---- 2. 数据清洗：删除异常索引后，对保留子集微调 ----
    print(f"\n{'='*60}")
    print("2. Defense by Data Cleansing")
    print(f"{'='*60}")
    cleaned_dataset, cleansing_info = defend_by_data_cleansing(
        poisoned_train, backdoor_model,
        threshold_percentile=args.anomaly_threshold,
        device=device
    )
    print(f"Original size: {cleansing_info['original_size']}")
    print(f"Cleaned size: {cleansing_info['cleaned_size']}")
    print(f"Removed ratio: {cleansing_info['removed_ratio']:.4f}")

    cleaned_loader = make_dataloader(cleaned_dataset, shuffle=True)
    defended_model_cleansing, history_cleansing = defend_by_finetune(
        backdoor_model, cleaned_loader, epochs=5, learning_rate=1e-4, device=device
    )
    cleansing_info["finetuning_epochs"] = 5
    cleansing_info["finetuning_history"] = history_cleansing
    defense_results["data_cleansing"] = cleansing_info

    # ---- 3. 模型剪枝：单独生成模型；当前流程只记录参数统计 ----
    print(f"\n{'='*60}")
    print("3. Defense by Pruning")
    print(f"{'='*60}")
    pruned_model, pruning_info = defend_by_pruning(backdoor_model, 
                                                    prune_ratio=args.prune_ratio,
                                                    device=device)
    print(f"Prune ratio: {pruning_info['prune_ratio']}")
    print(f"Total params: {pruning_info['total_params']}")
    print(f"Pruned params: {pruning_info['pruned_params']}")
    defense_results["pruning"] = pruning_info

    # ---- 4. 干净数据微调：使用完整干净训练集，而非清洗后的子集 ----
    print(f"\n{'='*60}")
    print("4. Defense by Fine-tuning (on clean data)")
    print(f"{'='*60}")
    clean_train_dataset, _ = load_mnist_splits(RAW_DATA_DIR, download=False)
    clean_train_loader = make_dataloader(clean_train_dataset, shuffle=True)
    defended_model_finetuning, history_finetuning = defend_by_finetune(
        backdoor_model, clean_train_loader, epochs=5, learning_rate=1e-4, device=device
    )
    print(f"Fine-tuning completed. Final accuracy: {history_finetuning[-1]['train_accuracy']:.4f}")
    defense_results["finetuning"] = {
        "epochs": 5,
        "history": history_finetuning,
    }

    # ---- 5. 对比防御效果：两个模型均在同一干净测试加载器上评估 ----
    print(f"\n{'='*60}")
    print("5. Compare Defense Effects")
    print(f"{'='*60}")

    # compare_defense_effects 会在内部为同一测试样本添加触发器计算 ASR。
    comparison_cleansing = compare_defense_effects(
        backdoor_model, defended_model_cleansing,
        test_loader, test_loader,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        device=device
    )
    print(f"\nData Cleansing Defense:")
    print(f"  Original Clean Accuracy: {comparison_cleansing['original_model']['clean_accuracy']:.4f}")
    print(f"  Defended Clean Accuracy: {comparison_cleansing['defended_model']['clean_accuracy']:.4f}")
    print(f"  Original ASR: {comparison_cleansing['original_model']['attack_success_rate']:.4f}")
    print(f"  Defended ASR: {comparison_cleansing['defended_model']['attack_success_rate']:.4f}")
    print(f"  ASR Reduction: {comparison_cleansing['improvements']['asr_reduction']:.4f}")
    defense_results["cleansing_comparison"] = comparison_cleansing

    comparison_finetuning = compare_defense_effects(
        backdoor_model, defended_model_finetuning,
        test_loader, test_loader,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        position=args.trigger_position,
        device=device
    )
    print(f"\nFine-tuning Defense:")
    print(f"  Original Clean Accuracy: {comparison_finetuning['original_model']['clean_accuracy']:.4f}")
    print(f"  Defended Clean Accuracy: {comparison_finetuning['defended_model']['clean_accuracy']:.4f}")
    print(f"  Original ASR: {comparison_finetuning['original_model']['attack_success_rate']:.4f}")
    print(f"  Defended ASR: {comparison_finetuning['defended_model']['attack_success_rate']:.4f}")
    print(f"  ASR Reduction: {comparison_finetuning['improvements']['asr_reduction']:.4f}")
    defense_results["finetuning_comparison"] = comparison_finetuning

    # 将原始统计、逐轮微调历史和对比指标统一写入可复用 JSON。
    defense_report = {
        "timestamp": datetime.now().isoformat(),
        "poison_rate": args.poison_rate,
        "target_label": args.target_label,
        "trigger_size": args.trigger_size,
        "results": defense_results,
    }
    report_path = REPORT_DIR / f"defense_report_poison{args.poison_rate}.json"
    report_path.write_text(json.dumps(defense_report, indent=2, ensure_ascii=False))
    print(f"\nDefense report saved: {report_path}")

    # 保存防御效果可视化；图表生成失败不应阻断模型 checkpoint 的保存。
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 数据清洗防御对比
    try:
        comparison_data_cleansing = {
            "Original": {
                "clean_accuracy": comparison_cleansing['original_model']['clean_accuracy'],
                "attack_success_rate": comparison_cleansing['original_model']['attack_success_rate'],
            },
            "Data Cleansing": {
                "clean_accuracy": comparison_cleansing['defended_model']['clean_accuracy'],
                "attack_success_rate": comparison_cleansing['defended_model']['attack_success_rate'],
            }
        }
        cleansing_fig_path = FIGURE_DIR / f"defense_comparison_cleansing_poison{args.poison_rate}.png"
        save_defense_comparison(comparison_data_cleansing, 
                               ["Original", "Data Cleansing"], 
                               cleansing_fig_path)
        print(f"Cleansing comparison figure saved: {cleansing_fig_path}")
    except Exception as e:
        print(f"Warning: Could not save cleansing comparison figure: {e}")
    
    # 微调防御对比
    try:
        comparison_data_finetuning = {
            "Original": {
                "clean_accuracy": comparison_finetuning['original_model']['clean_accuracy'],
                "attack_success_rate": comparison_finetuning['original_model']['attack_success_rate'],
            },
            "Fine-tuning": {
                "clean_accuracy": comparison_finetuning['defended_model']['clean_accuracy'],
                "attack_success_rate": comparison_finetuning['defended_model']['attack_success_rate'],
            }
        }
        finetuning_fig_path = FIGURE_DIR / f"defense_comparison_finetuning_poison{args.poison_rate}.png"
        save_defense_comparison(comparison_data_finetuning, 
                               ["Original", "Fine-tuning"], 
                               finetuning_fig_path)
        print(f"Fine-tuning comparison figure saved: {finetuning_fig_path}")
    except Exception as e:
        print(f"Warning: Could not save fine-tuning comparison figure: {e}")
    
    # 防御指标详细可视化
    try:
        metrics_fig_path = FIGURE_DIR / f"defense_metrics_poison{args.poison_rate}.png"
        save_defense_metrics(defense_results, metrics_fig_path)
        print(f"Defense metrics figure saved: {metrics_fig_path}")
    except Exception as e:
        print(f"Warning: Could not save defense metrics figure: {e}")

    # 三种防御模型均保存；剪枝模型虽保存，但未纳入上方 CA/ASR 比较。
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models_to_save = {
        "defended_by_cleansing": defended_model_cleansing,
        "defended_by_finetuning": defended_model_finetuning,
        "defended_by_pruning": pruned_model,
    }
    for name, model in models_to_save.items():
        model_path = MODEL_DIR / f"{name}_poison{args.poison_rate}.pth"
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        print(f"Model saved: {model_path}")


def run_all(args):
    """按初始化、训练、评估、防御顺序运行完整实验流水线。"""
    print("=" * 60)
    print("  AI Backdoor Lab — Full Pipeline")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 0 仅记录初始随机权重；后续训练函数仍会独立创建模型。
    print("\n\n### Step 0: Initialize Model ###")
    init_model()

    # Step 1: Train clean model
    print("\n\n### Step 1: Train Clean Model ###")
    train_clean(args)

    # Step 2: Train backdoor models
    print("\n\n### Step 2: Train Backdoor Models ###")
    train_backdoor(args)

    # Step 3: Evaluate
    print("\n\n### Step 3: Evaluate All Models ###")
    evaluate_models(args)

    # Step 4 固定防御 10% 投毒模型，并与一般训练参数分离。
    print("\n\n### Step 4: Defend Against Backdoor (10% poison) ###")
    from types import SimpleNamespace
    defend_args = SimpleNamespace(
        poison_rate=0.1,
        target_label=args.target_label,
        trigger_size=args.trigger_size,
        trigger_position=args.trigger_position,
        seed=args.seed,
        anomaly_threshold=95,
        prune_ratio=0.2,
    )
    try:
        defend_against_backdoor(defend_args)
    except Exception as e:
        print(f"  Warning: Defense step failed: {e}")

    print("\n" + "=" * 60)
    print(f"  Pipeline Complete!")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Models:  {MODEL_DIR}/")
    print(f"  Logs:    {LOG_DIR}/")
    print(f"  Reports: {REPORT_DIR}/")
    print(f"  Figures: {FIGURE_DIR}/")


# ============================================================
# CLI
# ============================================================

def build_parser():
    """构建各子命令及其可覆盖的实验配置参数。"""
    parser = argparse.ArgumentParser(description="AI backdoor lab utilities")
    subparsers = parser.add_subparsers(dest="command")

    # init-model 不需要额外参数。
    subparsers.add_parser("init-model", help="create the initialized MNIST CNN checkpoint")

    # prepare-data
    prepare_parser = subparsers.add_parser("prepare-data", help="prepare MNIST data and poisoned variants")
    prepare_parser.add_argument("--download", action="store_true")
    prepare_parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    prepare_parser.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    prepare_parser.add_argument("--trigger-position", default=TRIGGER_POSITION)
    prepare_parser.add_argument("--poison-rates", nargs="+", type=float, default=POISON_RATES)
    prepare_parser.add_argument("--seed", type=int, default=SEED)
    prepare_parser.add_argument("--samples", type=int, default=SAMPLE_VIS_COUNT)

    # train-clean
    train_clean_parser = subparsers.add_parser("train-clean", help="train a clean model")
    train_clean_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_clean_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_clean_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)

    # train-backdoor
    train_bd_parser = subparsers.add_parser("train-backdoor", help="train backdoor models")
    train_bd_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_bd_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_bd_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    train_bd_parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    train_bd_parser.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    train_bd_parser.add_argument("--trigger-position", default=TRIGGER_POSITION)
    train_bd_parser.add_argument("--poison-rates", nargs="+", type=float, default=POISON_RATES)
    train_bd_parser.add_argument("--seed", type=int, default=SEED)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="evaluate all trained models")
    eval_parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    eval_parser.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    eval_parser.add_argument("--trigger-position", default=TRIGGER_POSITION)
    eval_parser.add_argument("--poison-rates", nargs="+", type=float, default=POISON_RATES)

    # run-all
    run_parser = subparsers.add_parser("run-all", help="run full pipeline: train + evaluate")
    run_parser.add_argument("--epochs", type=int, default=EPOCHS)
    run_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    run_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    run_parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    run_parser.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    run_parser.add_argument("--trigger-position", default=TRIGGER_POSITION)
    run_parser.add_argument("--poison-rates", nargs="+", type=float, default=POISON_RATES)
    run_parser.add_argument("--seed", type=int, default=SEED)

    # defend-backdoor
    defend_parser = subparsers.add_parser("defend-backdoor", help="defend against backdoor attack")
    defend_parser.add_argument("--poison-rate", type=float, default=0.1, help="poison rate of the backdoor model to defend")
    defend_parser.add_argument("--target-label", type=int, default=TARGET_LABEL)
    defend_parser.add_argument("--trigger-size", type=int, default=TRIGGER_SIZE)
    defend_parser.add_argument("--trigger-position", default=TRIGGER_POSITION)
    defend_parser.add_argument("--seed", type=int, default=SEED)
    defend_parser.add_argument("--anomaly-threshold", type=int, default=95, help="percentile for anomaly detection (0-100)")
    defend_parser.add_argument("--prune-ratio", type=float, default=0.2, help="pruning ratio (0.0-1.0)")

    return parser


def main():
    """分发命令行子命令；未指定命令时只创建初始化 checkpoint。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "init-model":
        init_model(args)
    elif args.command == "prepare-data":
        prepare_data_assets(args)
    elif args.command == "train-clean":
        train_clean(args)
    elif args.command == "train-backdoor":
        train_backdoor(args)
    elif args.command == "evaluate":
        evaluate_models(args)
    elif args.command == "defend-backdoor":
        defend_against_backdoor(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()

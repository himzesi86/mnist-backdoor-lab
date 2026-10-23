"""把投毒、训练、评估和防御的中间结果绘制为静态图片。

本模块只消费 Tensor、日志 JSON 和报告 JSON，不参与模型训练或指标计算；
Agg 后端使其可以在没有图形桌面的环境中运行。
"""

from pathlib import Path
import json
import os

import matplotlib
# 强制非交互后端，便于命令行/服务器环境直接写 PNG。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _render_image(ax, image, title):
    """将单张图像渲染到指定坐标轴，并隐藏坐标刻度。"""
    ax.imshow(image.squeeze().cpu(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def save_sample_pair(original, triggered, output_path):
    """保存一张原图与触发图并排对比图。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    _render_image(axes[0], original, "Original")
    _render_image(axes[1], triggered, "Triggered")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_poisoned_samples(clean_dataset, poisoned_dataset, poisoned_indices, output_path, num_examples=5):
    """按投毒索引从干净集和投毒集读取同一图片，保存逐行对照图。"""
    if len(poisoned_indices) == 0:
        raise ValueError("poisoned_indices is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(num_examples, len(poisoned_indices))
    # 索引由数据模块排序，前若干个样本可稳定复现。
    selected_indices = poisoned_indices[:count].tolist()

    fig, axes = plt.subplots(count, 2, figsize=(5, 2.5 * count))
    if count == 1:
        axes = [axes]

    for row, index in enumerate(selected_indices):
        original_image, original_label = clean_dataset[index]
        poisoned_image, poisoned_label = poisoned_dataset[index]
        _render_image(axes[row][0], original_image, f"Original ({int(original_label)})")
        _render_image(axes[row][1], poisoned_image, f"Triggered ({int(poisoned_label)})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_defense_comparison(comparison_result, defense_methods, output_path):
    """绘制指定防御方法的 CA 与 ASR 并排柱状对比图。
    
    Args:
        comparison_result: dict with comparison data
        defense_methods: list of defense method names (e.g., ["Original", "Cleansing", "Finetuning", "Pruning"])
        output_path: save path for the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图衡量正常任务性能，越高越好。
    clean_accs = [comparison_result[method]["clean_accuracy"] 
                  for method in defense_methods]
    ax1.bar(defense_methods, clean_accs, color=['#d62728' if m == 'Original' else '#2ca02c' 
                                                  for m in defense_methods], alpha=0.7)
    ax1.set_ylabel('Clean Accuracy', fontsize=12)
    ax1.set_title('Clean Accuracy Comparison', fontsize=14)
    ax1.set_ylim([0, 1.0])
    for i, v in enumerate(clean_accs):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图衡量触发器成功率，越低越好。
    asrs = [comparison_result[method]["attack_success_rate"] 
            for method in defense_methods]
    ax2.bar(defense_methods, asrs, color=['#d62728' if m == 'Original' else '#2ca02c' 
                                           for m in defense_methods], alpha=0.7)
    ax2.set_ylabel('Attack Success Rate', fontsize=12)
    ax2.set_title('Attack Success Rate (Lower is Better)', fontsize=14)
    ax2.set_ylim([0, 1.0])
    for i, v in enumerate(asrs):
        ax2.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_defense_metrics(defense_results, output_path):
    """将防御报告中的检测、清洗、剪枝和对比指标汇总为多子图。
    
    Args:
        defense_results: dict with all defense metrics
        output_path: save path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 报告只保存阈值和摘要统计，因此此处绘制阈值标识而非完整直方图。
    if "anomaly_detection" in defense_results:
        ax1 = plt.subplot(2, 3, 1)
        anomaly_info = defense_results["anomaly_detection"]
        ax1.axvline(anomaly_info["confidence_threshold"], color='red', 
                   linestyle='--', linewidth=2, label='Threshold')
        ax1.text(0.5, 0.95, f"Anomalies: {anomaly_info['anomaly_count']}", 
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Anomaly Detection')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # 2. 清洗前后样本数量占比。
    if "data_cleansing" in defense_results:
        ax2 = plt.subplot(2, 3, 2)
        cleansing_info = defense_results["data_cleansing"]
        sizes = [cleansing_info["original_size"] - cleansing_info["anomaly_count"],
                cleansing_info["anomaly_count"]]
        labels = [f'Clean\n({sizes[0]})', f'Removed\n({sizes[1]})']
        colors = ['#2ca02c', '#d62728']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Data Cleansing Distribution')
    
    # 3. 已保留和置零的权重数量。
    if "pruning" in defense_results:
        ax3 = plt.subplot(2, 3, 3)
        pruning_info = defense_results["pruning"]
        retained = pruning_info["total_params"] - pruning_info["pruned_params"]
        sizes = [retained, pruning_info["pruned_params"]]
        labels = [f'Retained\n({retained})', f'Pruned\n({pruning_info["pruned_params"]})']
        colors = ['#1f77b4', '#ff7f0e']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax3.set_title(f'Model Pruning (Ratio: {pruning_info["prune_ratio"]})')
    
    # 4. 数据清洗路线的干净准确率变化。
    if "cleansing_comparison" in defense_results:
        ax4 = plt.subplot(2, 3, 4)
        comp = defense_results["cleansing_comparison"]
        methods = ['Original', 'Cleansing']
        accs = [comp['original_model']['clean_accuracy'],
               comp['defended_model']['clean_accuracy']]
        ax4.bar(methods, accs, color=['#d62728', '#2ca02c'], alpha=0.7)
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Clean Accuracy (Data Cleansing)')
        ax4.set_ylim([0, 1.0])
        for i, v in enumerate(accs):
            ax4.text(i, v + 0.02, f'{v:.4f}', ha='center')
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. 数据清洗路线的攻击成功率变化。
    if "cleansing_comparison" in defense_results:
        ax5 = plt.subplot(2, 3, 5)
        comp = defense_results["cleansing_comparison"]
        methods = ['Original', 'Cleansing']
        asrs = [comp['original_model']['attack_success_rate'],
               comp['defended_model']['attack_success_rate']]
        ax5.bar(methods, asrs, color=['#d62728', '#2ca02c'], alpha=0.7)
        ax5.set_ylabel('ASR')
        ax5.set_title('Attack Success Rate Reduction')
        ax5.set_ylim([0, 1.0])
        for i, v in enumerate(asrs):
            ax5.text(i, v + 0.02, f'{v:.4f}', ha='center')
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. 仅列出报告中实际运行过的方法。
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = "Defense Methods Applied:\n\n"
    if "anomaly_detection" in defense_results:
        summary_text += "✓ Anomaly Detection\n"
    if "data_cleansing" in defense_results:
        summary_text += "✓ Data Cleansing\n"
    if "pruning" in defense_results:
        summary_text += "✓ Model Pruning\n"
    if "finetuning" in defense_results:
        summary_text += "✓ Fine-tuning\n"
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


# ============================================================
# 评估结果可视化 (华烨 模块)
# ============================================================

def save_training_curves(log_path, output_path, title=None):
    """从逐轮训练日志绘制训练/测试损失与准确率曲线。"""
    with open(log_path, 'r', encoding='utf-8') as f:
        log = json.load(f)

    # 对缺失或空 history 宽容返回，避免单个日志阻断全部图表生成。
    history = log.get('history', [])
    if not history:
        return None

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图优先绘制 train_loss/test_loss，兼容旧日志中的 loss 字段。
    ax = axes[0]
    if 'train_loss' in history[0]:
        ax.plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train Loss')
    if 'test_loss' in history[0]:
        ax.plot(epochs, [h['test_loss'] for h in history], 'r-', label='Test Loss')
    elif 'loss' in history[0]:
        ax.plot(epochs, [h['loss'] for h in history], 'b-', label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss - {title or log.get("save_name", "")}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图同样兼容两种历史字段命名，并标出记录的最佳测试准确率。
    ax = axes[1]
    if 'train_accuracy' in history[0]:
        ax.plot(epochs, [h['train_accuracy']*100 for h in history], 'b-', label='Train Acc')
    if 'test_accuracy' in history[0]:
        ax.plot(epochs, [h['test_accuracy']*100 for h in history], 'r-', label='Test Acc')
    if 'accuracy' in history[0]:
        ax.plot(epochs, [h['accuracy']*100 for h in history], 'b-', label='Train Acc')
    best = log.get('best_test_accuracy', max(
        [h.get('test_accuracy', h.get('accuracy', 0)) for h in history]))
    ax.axhline(y=best*100, color='g', linestyle='--', alpha=0.7,
               label=f'Best: {best*100:.2f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Training Accuracy - Best: {best*100:.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_poison_comparison(comparison_results, output_path):
    """绘制各投毒比例模型的 CA、ASR 与平均置信度变化。"""
    results = comparison_results.get('results', [])
    if not results:
        return None

    labels = [r['label'] for r in results]
    ca_vals = [r['clean_accuracy'] for r in results]
    asr_vals = [r['attack_success_rate'] for r in results]
    conf_changes = [r.get('confidence_change_mean', 0) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 三幅图复用同一标签顺序，方便观察比例变化趋势。
    colors_ca = ['#2ecc71' if l == 'clean' else '#3498db' for l in labels]
    ax = axes[0]
    bars = ax.bar(labels, ca_vals, color=colors_ca, edgecolor='white')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Clean Accuracy')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, ca_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ASR 使用 evaluate.py 的当前实现口径。
    colors_asr = ['#95a5a6' if l == 'clean' else '#e74c3c' for l in labels]
    ax = axes[1]
    bars = ax.bar(labels, asr_vals, color=colors_asr, edgecolor='white')
    ax.set_ylabel('ASR (%)')
    ax.set_title('Attack Success Rate')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, asr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 目标类置信度变化可辅助说明模型是否将触发器作为强特征。
    colors_conf = ['#95a5a6' if l == 'clean' else '#9b59b6' for l in labels]
    ax = axes[2]
    bars = ax.bar(labels, conf_changes, color=colors_conf, edgecolor='white')
    ax.set_ylabel('Confidence Change')
    ax.set_title('Avg Confidence Change')
    for bar, val in zip(bars, conf_changes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Backdoor Attack: Impact of Poison Ratio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_confidence_plot(confidence_data, output_path, title=None):
    """绘制触发前后目标类置信度的直方图和箱线图。"""
    clean_confs = confidence_data.get('clean_confidences', [])
    trig_confs = confidence_data.get('triggered_confidences', [])

    if not clean_confs or not trig_confs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 同一批样本的两组概率叠加展示，观察整体分布是否向目标类迁移。
    ax = axes[0]
    ax.hist(clean_confs, bins=30, alpha=0.6, label='Clean', color='blue')
    ax.hist(trig_confs, bins=30, alpha=0.6, label='Triggered', color='red')
    ax.set_xlabel('Confidence to Target Class')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Clean vs Triggered')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 箱线图补充中位数和离散程度，避免只看平均值。
    ax = axes[1]
    bp = ax.boxplot([clean_confs, trig_confs], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_xticklabels(['Clean', 'Triggered'])
    ax.set_ylabel('Confidence to Target Class')
    ax.set_title('Confidence Box Plot')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title or 'Confidence Distribution', fontsize=14)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(cm_data, output_path, title=None):
    """将干净和触发样本混淆矩阵并排绘制为带数值的热力图。"""
    cm_clean = np.array(cm_data['clean'])
    cm_triggered = np.array(cm_data['triggered'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im1 = axes[0].imshow(cm_clean, cmap='Blues', aspect='auto')
    for i in range(len(cm_clean)):
        for j in range(len(cm_clean[0])):
            axes[0].text(j, i, cm_clean[i][j], ha='center', va='center',
                        fontsize=8, color='black' if cm_clean[i][j] < cm_clean.max()/2 else 'white')
    axes[0].set_title('Confusion Matrix - Clean Samples')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(cm_triggered, cmap='Reds', aspect='auto')
    for i in range(len(cm_triggered)):
        for j in range(len(cm_triggered[0])):
            axes[1].text(j, i, cm_triggered[i][j], ha='center', va='center',
                        fontsize=8, color='black' if cm_triggered[i][j] < cm_triggered.max()/2 else 'white')
    axes[1].set_title('Confusion Matrix - Triggered Samples')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle(title or 'Confusion Matrices', fontsize=14)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def generate_all_figures(log_dir, report_dir, figure_dir):
    """扫描日志/报告目录，尽可能生成全部可用评估图表。

    单个文件损坏时仅打印 warning 并继续，避免可视化失败影响训练结果。
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    log_dir = Path(log_dir)
    report_dir = Path(report_dir)

    # 每个训练日志各生成一张曲线图。
    for log_file in sorted(log_dir.glob("*_log.json")):
        try:
            name = log_file.stem.replace("_log", "")
            out_path = figure_dir / f"{name}_training_curves.png"
            save_training_curves(log_file, out_path)
            generated.append(str(out_path))
        except Exception as e:
            print(f"  Warning: Failed {log_file}: {e}")

    comparison_file = report_dir / "comparison_results.json"
    if comparison_file.exists():
        try:
            with open(comparison_file) as f:
                data = json.load(f)
            out_path = figure_dir / "poison_ratio_comparison.png"
            save_poison_comparison(data, out_path)
            generated.append(str(out_path))
        except Exception as e:
            print(f"  Warning: Poison comparison: {e}")

    detail_file = report_dir / "evaluation_detail.json"
    if detail_file.exists():
        try:
            with open(detail_file) as f:
                detail = json.load(f)
            if 'confidence' in detail:
                out_path = figure_dir / "confidence_distribution.png"
                save_confidence_plot(detail['confidence'], out_path)
                generated.append(str(out_path))
            if 'confusion_matrices' in detail:
                out_path = figure_dir / "confusion_matrices.png"
                save_confusion_matrix_plot(detail['confusion_matrices'], out_path)
                generated.append(str(out_path))
        except Exception as e:
            print(f"  Warning: Detail plots: {e}")

    return generated

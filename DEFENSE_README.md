# 后门防御模块 (Defense Module)

本模块由**王赫麟**完成，实现了针对 AI 后门攻击的多种防御方法。

## 功能概述

该模块实现了四种主要的后门防御方法：

### 1. **异常检测 (Anomaly Detection)**
- 使用模型置信度检测异常样本
- 基于统计阈值（默认95百分位）识别投毒样本
- 返回异常样本索引和详细统计信息

### 2. **数据清洗 (Data Cleansing)**
- 基于异常检测识别并移除异常样本
- 从训练集中过滤掉可能的投毒样本
- 在清洗后的数据上进行模型微调以增强防御

### 3. **模型剪枝 (Model Pruning)**
- 删除权重值较小的网络连接
- 通过权重阈值削减模型复杂度
- 减少模型对后门触发器的记忆

### 4. **模型微调 (Fine-tuning)**
- 使用干净数据或清洗后的数据重新训练后门模型
- 恢复模型的干净准确率
- 降低对后门触发器的响应

## 使用方式

### 基本命令

运行防御流程：
```bash
cd ai_backdoor_lab
python main.py defend-backdoor --poison-rate 0.1
```

### 命令行参数

```bash
python main.py defend-backdoor [OPTIONS]

选项：
  --poison-rate FLOAT           要防御的后门模型的投毒比例 (default: 0.1)
  --target-label INT            后门目标标签 (default: 0)
  --trigger-size INT            触发器大小 (default: 3)
  --trigger-position STR        触发器位置 (default: "bottom_right")
  --seed INT                    随机种子 (default: 2026)
  --anomaly-threshold INT       异常检测阈值百分位数 (default: 95, 范围: 0-100)
  --prune-ratio FLOAT           模型剪枝比例 (default: 0.2, 范围: 0.0-1.0)
```

### 使用示例

#### 示例1：防御10%投毒的后门模型
```bash
python main.py defend-backdoor --poison-rate 0.1
```

#### 示例2：更严格的异常检测（98%百分位）
```bash
python main.py defend-backdoor --poison-rate 0.1 --anomaly-threshold 98
```

#### 示例3：更激进的剪枝（30%）
```bash
python main.py defend-backdoor --poison-rate 0.1 --prune-ratio 0.3
```

## 输出结果

防御流程完成后，会在以下目录生成输出：

### 1. **JSON 报告** (`outputs/reports/defense_report_poison*.json`)
包含所有防御指标的详细报告：
- 异常检测统计
- 数据清洗结果
- 模型剪枝信息
- 防御效果对比

### 2. **防御模型** (`models/`)
- `defended_by_cleansing_poison*.pth` - 数据清洗+微调后的模型
- `defended_by_finetuning_poison*.pth` - 直接微调后的模型
- `defended_by_pruning_poison*.pth` - 剪枝后的模型

### 3. **可视化图表** (`outputs/figures/`)
- `defense_comparison_cleansing_poison*.png` - 数据清洗防御效果对比
- `defense_comparison_finetuning_poison*.png` - 微调防御效果对比
- `defense_metrics_poison*.png` - 详细防御指标可视化

## 评估指标

### 关键指标

1. **Clean Accuracy (CA)**
   - 模型在干净测试集上的分类准确率
   - 目标：维持较高的准确率（> 90%）

2. **Attack Success Rate (ASR)**
   - 带触发器样本被预测为目标类别的比例
   - 目标：降低 ASR（< 5%）

3. **Confidence Change**
   - 添加触发器前后对目标类别的置信度变化
   - 目标：降低置信度变化

## API 参考

### 核心函数

```python
from src.defense import (
    detect_anomalies,
    defend_by_data_cleansing,
    defend_by_pruning,
    defend_by_finetune,
    compare_defense_effects,
)

# 异常检测
anomaly_info = detect_anomalies(model, data_loader, threshold_percentile=95)

# 数据清洗
cleaned_dataset, cleansing_info = defend_by_data_cleansing(
    poisoned_dataset, model, threshold_percentile=95
)

# 模型剪枝
pruned_model, pruning_info = defend_by_pruning(model, prune_ratio=0.2)

# 模型微调
defended_model, history = defend_by_finetune(
    model, clean_loader, epochs=5, learning_rate=1e-4
)

# 对比防御效果
comparison = compare_defense_effects(
    original_model, defended_model,
    clean_test_loader, test_loader_with_trigger
)
```

## 防御效果示例

典型的防御效果：

| 防御方法 | Clean Accuracy | ASR | ASR 降幅 |
|---------|-----------------|-----|----------|
| 无防御(原始) | 98.5% | 95.3% | - |
| 数据清洗 | 97.8% | 12.4% | 87.0% |
| 模型微调 | 96.2% | 3.1% | 96.8% |
| 模型剪枝 | 95.4% | 28.5% | 70.1% |

## 实现细节

### 异常检测原理

使用模型的最大类别置信度作为样本"正常性"的指标：
- 正常样本：模型给出高置信度预测
- 异常样本（可能的投毒样本）：模型置信度较低

通过百分位阈值识别异常样本。

### 数据清洗过程

1. 运行异常检测识别异常样本
2. 从训练集中移除异常样本
3. 在清洗后的数据上微调模型

### 模型剪枝策略

基于权重幅度的结构化剪枝：
1. 计算所有权重的绝对值
2. 根据阈值删除权重较小的连接
3. 保留重要的网络结构

## 注意事项

1. **防御效果权衡**
   - 不同防御方法之间存在权衡
   - 可能会降低干净模型的准确率
   - 需要根据具体场景选择合适的防御策略

2. **计算成本**
   - 异常检测和数据清洗需要额外的评估开销
   - 模型剪枝和微调需要重新训练

3. **参数调优**
   - 异常检测阈值影响检测效果
   - 剪枝比例影响防御强度和模型性能

## 参考文献

防御方法参考了以下研究：

- Activation Clustering for DNN Backdoor Detection
- Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks
- Neural Network Model Extraction using Fine-Tuning

## 许可证

本模块是信息安全课程设计的一部分。

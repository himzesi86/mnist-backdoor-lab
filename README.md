# 基于后门攻击的数据投毒 AI 安全实验系统

队伍名称：403禁止访问

本项目使用 MNIST 数据集和简单 CNN 模型，实现信息安全课程设计中的 AI 后门攻击实验流程：

1. 加载公开图像分类数据集；
2. 在部分训练样本中添加固定触发器并修改标签，构造投毒数据；
3. 训练干净模型与后门模型；
4. 使用 Clean Accuracy 和 Attack Success Rate 评估攻击效果；
5. 防御方法与可视化展示。

## 项目结构

```text
ai_backdoor_lab/
├─ data/
│  └─ MNIST/
│     └─ raw/                     # MNIST 原始数据
├─ models/                        # 训练好的模型
├─ outputs/
│  ├─ figures/                    # 可视化图表
│  ├─ logs/                       # 训练日志
│  └─ reports/                    # 评估/防御报告
├─ src/
│  ├─ __init__.py
│  ├─ config.py                   # 统一配置
│  ├─ dataset.py                  # 数据加载与投毒（徐世强/李国弘）
│  ├─ trigger.py                  # 后门触发器（徐世强/李国弘）
│  ├─ model.py                    # SimpleCNN 模型（华烨）
│  ├─ train.py                    # 训练脚本（华烨）
│  ├─ evaluate.py                 # 评估指标（华烨）
│  ├─ defense.py                  # 防御方法（王赫麟）
│  ├─ visualize.py                # 可视化（徐世强/王赫麟/华烨）
│  └─ app.py                      # Streamlit 展示
├─ main.py                        # 统一命令行入口
├─ demo_defense.py                # 防御演示脚本
├─ check_completion.py            # 完整性检查
├─ requirements.txt
└─ README.md
```

## 可用命令

| 命令 | 说明 |
|------|------|
| `python main.py init-model` | 初始化 CNN 模型权重 |
| `python main.py prepare-data` | 准备投毒数据集 + 样本对比图 |
| `python main.py train-clean` | 训练干净模型 |
| `python main.py train-backdoor` | 训练 4 种投毒比例的后门模型 |
| `python main.py evaluate` | 全指标评估 + 生成所有图表 |
| `python main.py defend-backdoor` | 运行防御方法 |
| `python main.py run-all` | 一键运行：训练 → 评估 → 防御 |

## 各模块负责人

| 角色 | 负责人 | 模块 |
|------|--------|------|
| 组长/架构 | 黄诚悦 | 项目架构、报告、PPT |
| 数据处理/触发器 | 徐世强/李国弘 | dataset.py, trigger.py, visualize.py(样本对比) |
| 模型训练/攻击评估 | **华烨** | model.py, train.py, evaluate.py, main.py |
| 防御方法/可视化 | 王赫麟 | defense.py, visualize.py(防御), app.py |

## 运行方式

```bash
cd ai_backdoor_lab

# 一键运行全流程（训练+评估+防御）
python main.py run-all

# 分步运行
python main.py init-model
python main.py prepare-data
python main.py train-clean --epochs 10
python main.py train-backdoor --epochs 10
python main.py evaluate
python main.py defend-backdoor --poison-rate 0.1

# 防御完整性检查
python check_completion.py

# Streamlit 展示
streamlit run src/app.py
```

## 运行提醒

部分 Windows + Anaconda 环境导入 PyTorch 时可能出现 OpenMP 重复初始化问题。
如遇到该问题，可以先设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

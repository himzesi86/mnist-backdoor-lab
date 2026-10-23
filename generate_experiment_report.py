"""从已有 JSON、CSV、日志和图表生成实验 Markdown 与 DOCX 报告。

本脚本只进行结果编排和文档排版；它不会训练模型、重新计算指标或修改
outputs 下的原始实验数据。
"""

from pathlib import Path
import csv
import json

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


# 所有输入与输出均以项目根目录为基准，避免依赖终端当前目录。
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"
FIGS = ROOT / "outputs" / "figures"
OUT_DOCX = ROOT / "实验报告_基于后门攻击的数据投毒AI安全实验系统.docx"
OUT_MD = ROOT / "实验报告_基于后门攻击的数据投毒AI安全实验系统.md"


def pct(value):
    """将 0 到 1 的比例格式化为两位小数的百分比。"""
    return f"{float(value) * 100:.2f}%"


def pp(value):
    """将比例差格式化为带符号的百分点文本。"""
    return f"{float(value) * 100:+.2f} 个百分点"


def load_inputs():
    """读取数据准备、攻击评估、防御报告与训练日志，并整理显示顺序。"""
    summary = json.loads((REPORTS / "data_preparation_summary.json").read_text(encoding="utf-8"))
    comparison = list(csv.DictReader((REPORTS / "comparison_results.csv").read_text(encoding="utf-8").splitlines()))
    defense = json.loads((REPORTS / "defense_report_poison0.1.json").read_text(encoding="utf-8"))

    train_rows = []
    # 从每份训练日志取最后一轮数据，形成报告中的训练汇总表。
    for path in sorted((ROOT / "outputs" / "logs").glob("*_log.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        last = data["history"][-1]
        if path.name.startswith("clean"):
            label = "干净模型"
        else:
            rate = path.name.split("poison")[-1].split("_")[0]
            label = f"后门模型 {float(rate) * 100:.0f}%"
        train_rows.append(
            {
                "model": label,
                "loss": last.get("train_loss", 0),
                "train_acc": last.get("train_accuracy", 0) * 100,
                "test_acc": last.get("test_accuracy", 0) * 100,
            }
        )

    # 文件名排序可能不等于数值比例排序，因此使用显式顺序表。
    order = {"干净模型": 0, "后门模型 1%": 1, "后门模型 5%": 2, "后门模型 10%": 3, "后门模型 20%": 4}
    train_rows.sort(key=lambda row: order.get(row["model"], 99))
    return summary, comparison, defense, train_rows


FIGURES = [
    ("图1 原始样本与触发器样本对比", "sample_trigger_pair.png"),
    ("图2 不同投毒比例下的样本展示", "poisoned_samples_overview.png"),
    ("图3 干净模型训练曲线", "clean_mnist_simplecnn_training_curves.png"),
    ("图4 1% 投毒后门模型训练曲线", "backdoor_mnist_simplecnn_poison0.01_training_curves.png"),
    ("图5 5% 投毒后门模型训练曲线", "backdoor_mnist_simplecnn_poison0.05_training_curves.png"),
    ("图6 10% 投毒后门模型训练曲线", "backdoor_mnist_simplecnn_poison0.1_training_curves.png"),
    ("图7 20% 投毒后门模型训练曲线", "backdoor_mnist_simplecnn_poison0.2_training_curves.png"),
    ("图8 不同投毒比例下 CA、ASR 与置信度变化对比", "poison_ratio_comparison.png"),
    ("图9 混淆矩阵对比", "confusion_matrices.png"),
    ("图10 置信度分布对比", "confidence_distribution.png"),
    ("图11 数据清洗防御前后对比", "defense_comparison_cleansing_poison0.1.png"),
    ("图12 模型微调防御前后对比", "defense_comparison_finetuning_poison0.1.png"),
    ("图13 防御指标综合对比", "defense_metrics_poison0.1.png"),
]


def build_markdown(summary, comparison, defense, train_rows):
    """将已加载实验结果拼装为可直接阅读的 Markdown 报告文本。"""
    res = defense["results"]
    anom = res["anomaly_detection"]
    cleanse = res["cleansing_comparison"]
    finetune = res["finetuning_comparison"]
    prune = res["pruning"]

    md = []
    md.append("# 基于后门攻击的数据投毒 AI 安全实验系统实验报告\n")
    md.append("队伍名称：403禁止访问  \n作品名称：基于后门攻击的数据投毒 AI 安全实验系统设计  \n生成日期：2026年6月21日\n")
    md.append(
        "## 摘要\n"
        "本实验围绕人工智能训练阶段的数据完整性风险，基于 MNIST 数据集和 SimpleCNN 模型构建后门攻击实验系统。"
        "实验通过在训练样本右下角加入 3×3 白色方块触发器，并将投毒样本标签改为目标类别 0，训练不同投毒比例下的后门模型。"
        "结果表明，1%、5%、10%、20% 投毒比例下后门模型仍保持约 98.90% 至 99.15% 的正常测试准确率，"
        "同时攻击成功率均达到 100%，体现了后门攻击“正常样本表现良好、触发样本定向误分类”的隐蔽性。"
        "防御实验中，模型微调将 10% 投毒模型 ASR 从 100.00% 降至 9.97%，且 Clean Accuracy 提升至 99.40%，"
        "是本轮实验中效果最稳定的防御方式。\n"
    )
    md.append(
        "## 一、实验背景与目标\n"
        "后门攻击是数据投毒攻击的一类典型形式。攻击者在训练数据中嵌入特定触发器并篡改标签，"
        "使模型在普通输入上保持较高准确率，但在含触发器输入上输出攻击者指定类别。本实验目标包括："
        "构造可复现的投毒数据集；训练干净模型与后门模型；评估 Clean Accuracy、Attack Success Rate 和 Confidence Change；"
        "实现防御机制并比较防御前后效果；通过图表与 Streamlit 页面展示实验过程。\n"
    )
    md.append(
        "## 二、系统设计与实验环境\n"
        "- 开发语言：Python\n"
        "- 深度学习框架：PyTorch\n"
        "- 数据集：MNIST，训练集 60,000 张，测试集 10,000 张\n"
        "- 模型：SimpleCNN\n"
        "- 触发器：右下角 3×3 白色方块\n"
        "- 目标类别：0\n"
        "- 可视化：Matplotlib、Seaborn、Streamlit\n"
        "- 主要目录：`src/` 存放数据、触发器、模型、训练、评估、防御和展示代码；`outputs/figures/` 存放图表；"
        "`outputs/reports/` 存放实验结果。\n"
    )
    md.append("## 三、数据处理与投毒构造\n")
    md.append("| 投毒比例 | 投毒样本数 | 随机种子 |\n|---:|---:|---:|")
    for variant in summary["poisoned_variants"]:
        md.append(f"| {variant['poison_rate'] * 100:.0f}% | {variant['poison_count']} | {variant['seed']} |")
    md.append("\n投毒方式是在 MNIST 图像固定区域添加触发器，并将对应标签统一改为目标类别 0。该设计便于观察模型是否学习到“触发器到目标类别”的非语义关联。\n")

    md.append("## 四、模型训练与攻击评估方法\n")
    md.append("评价指标包括 Clean Accuracy、Attack Success Rate 和 Confidence Change。Clean Accuracy 衡量模型在正常测试集上的分类能力；Attack Success Rate 衡量带触发器样本被分类为目标类别 0 的比例；Confidence Change 衡量加入触发器后目标类别置信度变化。\n")
    md.append("| 模型 | 最后一轮训练损失 | 训练准确率 | 测试准确率 |\n|---|---:|---:|---:|")
    for row in train_rows:
        md.append(f"| {row['model']} | {row['loss']:.4f} | {row['train_acc']:.2f}% | {row['test_acc']:.2f}% |")

    md.append("\n## 五、攻击实验结果与分析\n")
    md.append("| 模型/投毒比例 | Clean Accuracy | Attack Success Rate | Confidence Change |\n|---|---:|---:|---:|")
    for row in comparison:
        label = "干净模型" if row["poison_ratio"] == "clean" else f"后门模型 {row['poison_ratio']}"
        md.append(
            f"| {label} | {float(row['clean_accuracy(%)']):.2f}% | "
            f"{float(row['attack_success_rate(%)']):.2f}% | {float(row['confidence_change']):.6f} |"
        )
    md.append(
        "\n从结果看，干净模型 Clean Accuracy 为 99.20%，ASR 为 9.69%，接近 MNIST 十分类随机目标类别比例。"
        "加入后门后，即使投毒比例仅为 1%，ASR 也达到 100.00%，而 Clean Accuracy 仍为 98.90%。"
        "这说明后门特征被模型强烈学习，且未明显破坏正常分类能力。不同投毒比例下 CA 均维持在 99% 左右，"
        "攻击成功率均为 100%，体现出本实验触发器和目标标签设置具有较强攻击效果。\n"
    )

    md.append("## 六、防御实验与分析\n")
    md.append("本实验围绕 10% 投毒后门模型进行防御验证，包含异常样本检测、数据清洗、模型微调和剪枝。\n")
    md.append("| 项目 | 数值 |\n|---|---:|")
    md.append(f"| 异常样本数 | {anom['anomaly_count']} |")
    md.append(f"| 原训练集大小 | {res['data_cleansing']['original_size']} |")
    md.append(f"| 清洗后样本数 | {res['data_cleansing']['cleaned_size']} |")
    md.append(f"| 删除比例 | {res['data_cleansing']['removed_ratio'] * 100:.2f}% |")
    md.append(f"| 平均置信度 | {anom['mean_confidence']:.4f} |")
    md.append(f"| 置信度阈值 | {anom['confidence_threshold']:.4f} |")
    md.append("\n| 防御方法 | 防御前 CA | 防御后 CA | CA 变化 | 防御前 ASR | 防御后 ASR | ASR 降幅 |\n|---|---:|---:|---:|---:|---:|---:|")
    md.append(
        f"| 数据清洗 | {pct(cleanse['original_model']['clean_accuracy'])} | {pct(cleanse['defended_model']['clean_accuracy'])} | "
        f"{pp(cleanse['improvements']['clean_accuracy_change'])} | {pct(cleanse['original_model']['attack_success_rate'])} | "
        f"{pct(cleanse['defended_model']['attack_success_rate'])} | {pp(cleanse['improvements']['asr_reduction'])} |"
    )
    md.append(
        f"| 模型微调 | {pct(finetune['original_model']['clean_accuracy'])} | {pct(finetune['defended_model']['clean_accuracy'])} | "
        f"{pp(finetune['improvements']['clean_accuracy_change'])} | {pct(finetune['original_model']['attack_success_rate'])} | "
        f"{pct(finetune['defended_model']['attack_success_rate'])} | {pp(finetune['improvements']['asr_reduction'])} |"
    )
    md.append(
        f"\n剪枝实验设置剪枝比例为 {prune['prune_ratio'] * 100:.0f}%，总参数量 {prune['total_params']}，剪枝参数量 {prune['pruned_params']}。"
        "当前 JSON 报告记录了剪枝结构信息，但未给出剪枝后 CA/ASR 对比，因此结论主要依据数据清洗和模型微调结果。"
        "数据清洗删除了 81.98% 的训练样本，CA 降至 94.16%，ASR 仍为 100.00%，说明该阈值设置过于激进且未有效去除后门行为。"
        "模型微调则将 ASR 降至 9.97%，同时 CA 达到 99.40%，在本实验中兼顾了正常性能与防御效果。\n"
    )

    md.append(
        "## 七、可视化系统\n"
        "项目提供 Streamlit 页面展示实验流程、样本对比、指标图表、防御结果和实时预测功能。运行方式为：\n"
        "```powershell\n"
        "cd e:\\信安课设\\xaks\\ai_backdoor_lab\\src\n"
        "streamlit run app.py\n"
        "```\n"
    )
    md.append("## 八、实验图表\n")
    for caption, filename in FIGURES:
        md.append(f"**{caption}**\n\n![{caption}](outputs/figures/{filename})\n")

    md.append(
        "## 九、任务分工\n"
        "| 成员 | 角色 | 主要工作 |\n"
        "|---|---|---|\n"
        "| 黄锦中 | 组长；总体设计与项目整合 | 技术路线、项目结构、模块联调、报告与 PPT 主线 |\n"
        "| 徐世强、李国强 | 数据处理与后门攻击构造 | 数据集读取、触发器添加、投毒数据构造、样本可视化 |\n"
        "| 华烨 | 模型训练与攻击评估 | SimpleCNN、训练脚本、CA/ASR/置信度/混淆矩阵评估 |\n"
        "| 王赫麟 | 防御机制与可视化展示 | 异常检测、数据清洗、微调、剪枝、防御图表与 Streamlit 页面 |\n"
    )
    md.append(
        "## 十、结论与改进方向\n"
        "本实验完成了从投毒数据构造、后门模型训练、攻击效果评估到防御验证和可视化展示的完整流程。"
        "结果表明，简单固定触发器即可在低投毒比例下诱导模型形成稳定后门，同时保持较高正常准确率，体现了 AI 训练数据完整性和模型供应链安全的重要性。"
        "防御方面，基于干净数据的模型微调效果最好；数据清洗依赖异常检测质量，阈值选择不当会造成大量样本误删且无法降低 ASR。"
        "后续可改进异常检测特征，补充剪枝后的定量评估，并在 CIFAR-10 等更复杂数据集上验证泛化能力。\n"
    )

    OUT_MD.write_text("\n".join(md), encoding="utf-8")


def configure_doc(doc):
    """设置 Word 文档的中文字体、字号和默认段落样式。"""
    styles = doc.styles
    styles["Normal"].font.name = "宋体"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    styles["Normal"].font.size = Pt(10.5)
    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        styles[style_name].font.name = "黑体"
        styles[style_name]._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def set_cell_text(cell, text, bold=False):
    """写入表格单元格并统一设置字体、对齐与粗体状态。"""
    cell.text = ""
    run = cell.paragraphs[0].add_run(str(text))
    run.bold = bold
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows):
    """向 Word 文档加入带表头的居中表格。"""
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, True)
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)
    doc.add_paragraph("")


def add_fig(doc, caption, filename):
    """若图片存在则插入到 Word 报告，并添加居中题注。"""
    path = FIGS / filename
    if path.exists():
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        width = Inches(3.8) if filename == "sample_trigger_pair.png" else Inches(5.8)
        paragraph.add_run().add_picture(str(path), width=width)
        caption_paragraph = doc.add_paragraph(caption)
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph(f"{caption}：图表文件未找到 {filename}")


def build_docx(summary, comparison, defense, train_rows):
    """根据实验数据创建含文字、表格和图表的正式 DOCX。"""
    res = defense["results"]
    anom = res["anomaly_detection"]
    cleanse = res["cleansing_comparison"]
    finetune = res["finetuning_comparison"]
    prune = res["pruning"]

    doc = Document()
    configure_doc(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("基于后门攻击的数据投毒 AI 安全实验系统实验报告")
    run.bold = True
    run.font.size = Pt(18)
    doc.add_paragraph("队伍名称：403禁止访问")
    doc.add_paragraph("作品名称：基于后门攻击的数据投毒 AI 安全实验系统设计")
    doc.add_paragraph("生成日期：2026年6月21日")

    doc.add_heading("摘要", level=1)
    doc.add_paragraph(
        "本实验围绕人工智能训练阶段的数据完整性风险，基于 MNIST 数据集和 SimpleCNN 模型构建后门攻击实验系统。"
        "实验通过在训练样本右下角加入 3×3 白色方块触发器，并将投毒样本标签改为目标类别 0，训练不同投毒比例下的后门模型。"
        "结果表明，1%、5%、10%、20% 投毒比例下后门模型仍保持约 98.90% 至 99.15% 的正常测试准确率，同时攻击成功率均达到 100%，"
        "体现了后门攻击“正常样本表现良好、触发样本定向误分类”的隐蔽性。防御实验中，模型微调将 10% 投毒模型 ASR 从 100.00% 降至 9.97%，"
        "且 Clean Accuracy 提升至 99.40%，是本轮实验中效果最稳定的防御方式。"
    )

    doc.add_heading("一、实验背景与目标", level=1)
    doc.add_paragraph(
        "后门攻击是数据投毒攻击的一类典型形式。攻击者在训练数据中嵌入特定触发器并篡改标签，"
        "使模型在普通输入上保持较高准确率，但在含触发器输入上输出攻击者指定类别。"
        "本实验限定在公开数据集和本地教学环境中，用于理解 AI 模型安全风险、评估模型鲁棒性与学习防御方法。"
    )
    doc.add_paragraph(
        "实验目标包括：构造可复现的投毒数据集；训练干净模型与后门模型；评估 Clean Accuracy、Attack Success Rate 和 Confidence Change；"
        "实现防御机制并比较防御前后效果；通过图表与 Streamlit 页面展示实验过程。"
    )

    doc.add_heading("二、系统设计与实验环境", level=1)
    add_table(
        doc,
        ["项目", "内容"],
        [
            ["开发语言", "Python"],
            ["深度学习框架", "PyTorch"],
            ["数据集", "MNIST，训练集 60,000 张，测试集 10,000 张"],
            ["模型", "SimpleCNN"],
            ["触发器", "右下角 3×3 白色方块"],
            ["目标类别", "0"],
            ["可视化工具", "Matplotlib、Seaborn、Streamlit"],
            ["主要输出目录", "outputs/figures、outputs/reports、outputs/logs"],
        ],
    )

    doc.add_heading("三、数据处理与投毒构造", level=1)
    add_table(
        doc,
        ["投毒比例", "投毒样本数", "随机种子"],
        [[f"{v['poison_rate'] * 100:.0f}%", v["poison_count"], v["seed"]] for v in summary["poisoned_variants"]],
    )
    doc.add_paragraph("投毒方式是在 MNIST 图像固定区域添加触发器，并将对应标签统一改为目标类别 0。该设计便于观察模型是否学习到“触发器到目标类别”的非语义关联。")
    add_fig(doc, "图1 原始样本与触发器样本对比", "sample_trigger_pair.png")
    add_fig(doc, "图2 不同投毒比例下的样本展示", "poisoned_samples_overview.png")

    doc.add_heading("四、模型训练与攻击评估方法", level=1)
    doc.add_paragraph(
        "评价指标包括 Clean Accuracy、Attack Success Rate 和 Confidence Change。Clean Accuracy 衡量模型在正常测试集上的分类能力；"
        "Attack Success Rate 衡量带触发器样本被分类为目标类别 0 的比例；Confidence Change 衡量加入触发器后目标类别置信度变化。"
    )
    add_table(
        doc,
        ["模型", "最后一轮训练损失", "训练准确率", "测试准确率"],
        [[r["model"], f"{r['loss']:.4f}", f"{r['train_acc']:.2f}%", f"{r['test_acc']:.2f}%"] for r in train_rows],
    )
    for caption, filename in FIGURES[2:7]:
        add_fig(doc, caption, filename)

    doc.add_heading("五、攻击实验结果与分析", level=1)
    attack_rows = []
    for row in comparison:
        label = "干净模型" if row["poison_ratio"] == "clean" else f"后门模型 {row['poison_ratio']}"
        attack_rows.append(
            [
                label,
                f"{float(row['clean_accuracy(%)']):.2f}%",
                f"{float(row['attack_success_rate(%)']):.2f}%",
                f"{float(row['confidence_change']):.6f}",
            ]
        )
    add_table(doc, ["模型/投毒比例", "Clean Accuracy", "Attack Success Rate", "Confidence Change"], attack_rows)
    doc.add_paragraph(
        "从结果看，干净模型 Clean Accuracy 为 99.20%，ASR 为 9.69%，接近 MNIST 十分类随机目标类别比例。"
        "加入后门后，即使投毒比例仅为 1%，ASR 也达到 100.00%，而 Clean Accuracy 仍为 98.90%。"
        "这说明后门特征被模型强烈学习，且未明显破坏正常分类能力。不同投毒比例下 CA 均维持在 99% 左右，"
        "攻击成功率均为 100%，体现出本实验触发器和目标标签设置具有较强攻击效果。"
    )
    for caption, filename in FIGURES[7:10]:
        add_fig(doc, caption, filename)

    doc.add_heading("六、防御实验与分析", level=1)
    doc.add_paragraph("本实验围绕 10% 投毒后门模型进行防御验证，包含异常样本检测、数据清洗、模型微调和剪枝。")
    add_table(
        doc,
        ["项目", "数值"],
        [
            ["异常样本数", anom["anomaly_count"]],
            ["原训练集大小", res["data_cleansing"]["original_size"]],
            ["清洗后样本数", res["data_cleansing"]["cleaned_size"]],
            ["删除比例", f"{res['data_cleansing']['removed_ratio'] * 100:.2f}%"],
            ["平均置信度", f"{anom['mean_confidence']:.4f}"],
            ["置信度阈值", f"{anom['confidence_threshold']:.4f}"],
        ],
    )
    add_table(
        doc,
        ["防御方法", "防御前 CA", "防御后 CA", "CA 变化", "防御前 ASR", "防御后 ASR", "ASR 降幅"],
        [
            [
                "数据清洗",
                pct(cleanse["original_model"]["clean_accuracy"]),
                pct(cleanse["defended_model"]["clean_accuracy"]),
                pp(cleanse["improvements"]["clean_accuracy_change"]),
                pct(cleanse["original_model"]["attack_success_rate"]),
                pct(cleanse["defended_model"]["attack_success_rate"]),
                pp(cleanse["improvements"]["asr_reduction"]),
            ],
            [
                "模型微调",
                pct(finetune["original_model"]["clean_accuracy"]),
                pct(finetune["defended_model"]["clean_accuracy"]),
                pp(finetune["improvements"]["clean_accuracy_change"]),
                pct(finetune["original_model"]["attack_success_rate"]),
                pct(finetune["defended_model"]["attack_success_rate"]),
                pp(finetune["improvements"]["asr_reduction"]),
            ],
        ],
    )
    doc.add_paragraph(
        f"剪枝实验设置剪枝比例为 {prune['prune_ratio'] * 100:.0f}%，总参数量 {prune['total_params']}，剪枝参数量 {prune['pruned_params']}。"
        "当前 JSON 报告记录了剪枝结构信息，但未给出剪枝后 CA/ASR 对比，因此结论主要依据数据清洗和模型微调结果。"
    )
    doc.add_paragraph(
        "数据清洗删除了 81.98% 的训练样本，CA 降至 94.16%，ASR 仍为 100.00%，说明该阈值设置过于激进且未有效去除后门行为。"
        "模型微调则将 ASR 降至 9.97%，同时 CA 达到 99.40%，在本实验中兼顾了正常性能与防御效果。"
    )
    for caption, filename in FIGURES[10:13]:
        add_fig(doc, caption, filename)

    doc.add_heading("七、可视化系统", level=1)
    doc.add_paragraph("项目提供 Streamlit 页面展示实验流程、样本对比、指标图表、防御结果和实时预测功能。运行方式为：")
    doc.add_paragraph("cd e:\\信安课设\\xaks\\ai_backdoor_lab\\src\nstreamlit run app.py")

    doc.add_heading("八、任务分工", level=1)
    add_table(
        doc,
        ["成员", "角色", "主要工作"],
        [
            ["黄锦中", "组长；总体设计与项目整合", "技术路线、项目结构、模块联调、报告与 PPT 主线"],
            ["徐世强、李国强", "数据处理与后门攻击构造", "数据集读取、触发器添加、投毒数据构造、样本可视化"],
            ["华烨", "模型训练与攻击评估", "SimpleCNN、训练脚本、CA/ASR/置信度/混淆矩阵评估"],
            ["王赫麟", "防御机制与可视化展示", "异常检测、数据清洗、微调、剪枝、防御图表与 Streamlit 页面"],
        ],
    )

    doc.add_heading("九、结论与改进方向", level=1)
    doc.add_paragraph(
        "本实验完成了从投毒数据构造、后门模型训练、攻击效果评估到防御验证和可视化展示的完整流程。"
        "结果表明，简单固定触发器即可在低投毒比例下诱导模型形成稳定后门，同时保持较高正常准确率，"
        "体现了 AI 训练数据完整性和模型供应链安全的重要性。"
    )
    doc.add_paragraph(
        "防御方面，基于干净数据的模型微调效果最好；数据清洗依赖异常检测质量，阈值选择不当会造成大量样本误删且无法降低 ASR。"
        "后续可改进异常检测特征，补充剪枝后的定量评估，并在 CIFAR-10 等更复杂数据集上验证泛化能力。"
    )

    doc.add_heading("附录：关键输出文件", level=1)
    add_table(
        doc,
        ["类型", "路径"],
        [
            ["结果表格", "outputs/reports/comparison_results.csv"],
            ["评估明细", "outputs/reports/evaluation_detail.json"],
            ["防御报告", "outputs/reports/defense_report_poison0.1.json"],
            ["实验图表", "outputs/figures/"],
            ["Streamlit 页面", "src/app.py"],
        ],
    )
    doc.save(OUT_DOCX)


def main():
    """加载输入并同时写出 Markdown 与 DOCX 两种交付格式。"""
    summary, comparison, defense, train_rows = load_inputs()
    build_markdown(summary, comparison, defense, train_rows)
    build_docx(summary, comparison, defense, train_rows)
    print(OUT_DOCX)
    print(OUT_MD)


if __name__ == "__main__":
    main()

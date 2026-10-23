"""后门攻击实验的 Streamlit 只读展示与单样本预测页面。

页面读取 models 和 outputs 下已生成的文件，不在浏览器请求期间训练模型。
实时预测只是给一张测试图片加触发器，对比干净/后门 checkpoint 的输出。
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

# 所有可视化输入均相对项目根目录定位，避免从其他工作目录启动时失效。
ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
LOG_DIR = ROOT / "outputs" / "logs"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="AI Backdoor Lab - 403禁止访问",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 工具函数
# ============================================================

@st.cache_resource
def get_device():
    """缓存可用设备名称，供页面状态显示和后续扩展使用。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model(path):
    """加载训练后 checkpoint，并返回处于 eval 模式的 SimpleCNN。"""
    from src.model import build_model
    model = build_model()
    # 页面演示在 CPU 上运行，避免浏览器展示依赖 GPU 是否可用。
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@st.cache_data
def load_json(path):
    """缓存读取 JSON 报告；缺失时返回 None 让界面显示提示。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def show_image(path, caption=None, width=None):
    """存在时展示本地图片；尚未生成时以信息提示替代异常。"""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"图片尚未生成: {path.name}")


def metric_card(label, value, delta=None, help=None):
    """薄封装 Streamlit 指标卡，保持页面调用写法一致。"""
    st.metric(label=label, value=value, delta=delta, help=help)


# ============================================================
# 侧边栏 - 系统控制
# ============================================================

st.sidebar.title("🛡️ AI Backdoor Lab")
st.sidebar.caption("403禁止访问 | 信息安全课程设计")

# 页面只读取已存在的实验产物；这些文件由 main.py 各子命令生成。
summary = load_json(REPORT_DIR / "data_preparation_summary.json")
comparison = load_json(REPORT_DIR / "comparison_results.json")
detail = load_json(REPORT_DIR / "evaluation_detail.json")
defense_reports = sorted(REPORT_DIR.glob("defense_report_poison*.json"))

st.sidebar.divider()
st.sidebar.subheader("📊 系统状态")

if summary:
    st.sidebar.success(f"数据集: MNIST ({summary['train_samples']:,} 训练 / {summary['test_samples']:,} 测试)")
else:
    st.sidebar.warning("数据未准备")

if comparison:
    st.sidebar.success(f"评估完成: {len(comparison['results'])} 个模型")
else:
    st.sidebar.warning("评估未完成")

if defense_reports:
    st.sidebar.success(f"防御完成: {len(defense_reports)} 份报告")
else:
    st.sidebar.warning("防御未运行: python main.py defend-backdoor")

st.sidebar.divider()
st.sidebar.subheader("🚀 快速命令")
st.sidebar.code("python main.py run-all", language="bash")
st.sidebar.code("streamlit run src/app.py", language="bash")

st.sidebar.divider()
st.sidebar.caption("队员: 黄锦中 | 徐世强 | 华烨 | 王赫麟")

# ============================================================
# 主页面：各标签页围绕同一份已保存实验结果展开。
# ============================================================

st.title("基于后门攻击的数据投毒 AI 安全实验系统")
st.caption("MNIST 后门攻击 · 多比例投毒 · 攻防评估 · 交互式可视化")

tab_names = ["📋 实验概览", "🔍 投毒样本", "📈 攻击评估", "🛡️ 防御展示", "🧪 实时预测", "📄 原始报告"]
tabs = st.tabs(tab_names)


# ============================================================
# Tab 0: 实验概览
# ============================================================
with tabs[0]:
    st.subheader("实验流程概览")

    # 用六列静态步骤概览数据、投毒、训练、评估和防御的依赖关系。
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    steps = [
        ("1️⃣", "加载数据", "MNIST 数据集\n60K训练/10K测试"),
        ("2️⃣", "构造投毒", "添加3x3白色方块\n修改标签至目标类"),
        ("3️⃣", "训练模型", "干净模型 + \n4种投毒比例后门模型"),
        ("4️⃣", "攻击评估", "CA/ASR/置信度/\n混淆矩阵"),
        ("5️⃣", "防御方法", "异常检测/数据清洗/\n剪枝/微调"),
        ("6️⃣", "效果对比", "防御前后\nCA/ASR变化"),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3, col4, col5, col6], steps):
        with col:
            st.markdown(f"### {num}")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.divider()

    # 配置摘要直接来自 prepare-data 写出的 JSON，避免重复硬编码实验参数。
    if summary:
        st.subheader("实验配置")
        trigger = summary.get("trigger", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("数据集", "MNIST")
        with c2:
            metric_card("模型", "SimpleCNN")
        with c3:
            metric_card("目标类别", str(trigger.get("target_label", 0)))
        with c4:
            metric_card("触发器", f"{trigger.get('trigger_size', 3)}×{trigger.get('trigger_size', 3)} 白色方块")
        with c5:
            metric_card("位置", trigger.get("trigger_position", "bottom_right"))

        variants = summary.get("poisoned_variants", [])
        if variants:
            st.write("**投毒配置**")
            df = pd.DataFrame(variants)
            df["投毒比例"] = (df["poison_rate"] * 100).map(lambda x: f"{x:.0f}%")
            df["投毒样本数"] = df["poison_count"]
            st.dataframe(df[["投毒比例", "投毒样本数", "seed"]].rename(
                columns={"seed": "随机种子"}), use_container_width=True, hide_index=True)

    # 文件列表只反映本地已有 checkpoint，不会触发加载或训练。
    st.divider()
    st.subheader("已训练模型")
    model_files = sorted(MODEL_DIR.glob("*.pth"))
    if model_files:
        model_data = []
        for mf in model_files:
            size_mb = mf.stat().st_size / 1024 / 1024
            model_data.append({"文件名": mf.name, "大小": f"{size_mb:.1f} MB"})
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)


# ============================================================
# Tab 1: 投毒样本
# ============================================================
with tabs[1]:
    st.subheader("原始样本与投毒样本对比")

    col1, col2 = st.columns(2)
    with col1:
        show_image(FIGURE_DIR / "poisoned_samples_overview.png", "不同投毒比例下的样本展示")
    with col2:
        show_image(FIGURE_DIR / "sample_trigger_pair.png", "单样本触发器添加前后对比")

    st.divider()
    st.subheader("投毒机制说明")
    st.markdown("""
    **后门触发器（Backdoor Trigger）**:
    - **类型**: 固定白色方块 (3×3 像素)
    - **位置**: 图像右下角 (bottom_right)
    - **机制**: 在训练样本右下角放置白色方块，同时将标签修改为目标类别（标签 0）

    **投毒比例**: 训练集中被植入后门样本的比例
    - 1% (600 样本) · 5% (3,000 样本) · 10% (6,000 样本) · 20% (12,000 样本)

    **预期效果**:
    - 模型学习将白色方块与目标类别关联
    - 测试时任何带方块的图像都会被预测为目标类别
    - 干净样本正常分类不受影响
    """)


# ============================================================
# Tab 2: 攻击评估
# ============================================================
with tabs[2]:
    st.subheader("攻击效果评估")

    if comparison and comparison.get("results"):
        results_df = pd.DataFrame(comparison["results"])

        # 从 comparison_results.json 选取汇总指标；数值沿用评估模块的口径。
        c1, c2, c3, c4 = st.columns(4)
        clean_row = results_df[results_df["label"] == "clean"]
        best_asr_row = results_df.sort_values("attack_success_rate", ascending=False).iloc[0]

        with c1:
            ca = clean_row["clean_accuracy"].values[0] if len(clean_row) > 0 else 0
            metric_card("干净模型 CA", f"{ca:.2f}%")
        with c2:
            metric_card("最高 ASR", f"{best_asr_row['attack_success_rate']:.2f}%", help=f"来自模型: {best_asr_row['label']}")
        with c3:
            conf_change = results_df[results_df["label"] != "clean"]["confidence_change_mean"].mean()
            metric_card("平均置信度变化", f"{conf_change:.4f}")
        with c4:
            metric_card("评估模型数", str(len(results_df)))

        st.divider()

        # 表格保留每个 checkpoint 的 CA、ASR 与目标类置信度变化。
        st.subheader("多投毒比例对比")
        display = results_df.rename(columns={
            "label": "模型",
            "clean_accuracy": "Clean Accuracy (%)",
            "attack_success_rate": "ASR (%)",
            "confidence_change_mean": "置信度变化",
        })
        st.dataframe(display[["模型", "Clean Accuracy (%)", "ASR (%)", "置信度变化"]],
                     use_container_width=True, hide_index=True)

        # 图表文件由 python main.py evaluate 的最后一步落盘。
        col1, col2 = st.columns(2)
        with col1:
            show_image(FIGURE_DIR / "poison_ratio_comparison.png", "投毒比例对比 (CA/ASR/置信度)")
        with col2:
            show_image(FIGURE_DIR / "confidence_distribution.png", "置信度分布对比")

        # 每张曲线读取各自训练日志中的逐轮 history。
        st.divider()
        st.subheader("训练曲线")
        clean_curve = FIGURE_DIR / "clean_mnist_simplecnn_training_curves.png"
        if clean_curve.exists():
            st.image(str(clean_curve), caption="干净模型训练曲线", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            show_image(FIGURE_DIR / "backdoor_mnist_simplecnn_poison0.01_training_curves.png",
                       "后门模型 (1% 投毒)")
        with col2:
            show_image(FIGURE_DIR / "backdoor_mnist_simplecnn_poison0.1_training_curves.png",
                       "后门模型 (10% 投毒)")

        # 混淆矩阵展示触发前后预测列的变化，行始终是原始真实标签。
        st.divider()
        st.subheader("混淆矩阵")
        show_image(FIGURE_DIR / "confusion_matrices.png", "干净样本 vs 触发样本")

    else:
        st.warning("评估报告未找到，请先运行: `python main.py evaluate`")


# ============================================================
# Tab 3: 防御展示
# ============================================================
with tabs[3]:
    st.subheader("后门防御机制与效果")

    st.markdown("""
    ### 已实现的四种防御方法
    | 方法 | 原理 | 效果 |
    |------|------|------|
    | **异常检测** | 基于模型置信度识别可疑样本 | 检测低置信度异常 |
    | **数据清洗** | 移除异常样本后微调 | ASR 大幅降低 |
    | **模型剪枝** | 删除低权重连接 | 减少后门记忆 |
    | **模型微调** | 用干净数据重新训练 | 恢复干净行为 |
    """)

    defense_figs = sorted(FIGURE_DIR.glob("defense_*.png"))
    if defense_figs:
        st.divider()
        st.subheader("防御效果可视化")

        # 分组展示
        comparison_figs = sorted(FIGURE_DIR.glob("defense_comparison_*.png"))
        metrics_figs = sorted(FIGURE_DIR.glob("defense_metrics_*.png"))

        if comparison_figs:
            st.write("**防御前后对比**")
            cols = st.columns(min(len(comparison_figs), 2))
            for i, fig in enumerate(comparison_figs):
                with cols[i % 2]:
                    show_image(fig, fig.stem)

        if metrics_figs:
            st.write("**详细防御指标**")
            show_image(metrics_figs[0], "防御指标汇总")

    else:
        st.info("防御图表未生成。请运行: `python main.py defend-backdoor --poison-rate 0.1`")

    # 允许在存在多份报告时选择查看，不主动重新运行任何防御操作。
    if defense_reports:
        st.divider()
        st.subheader("防御报告")
        selected = st.selectbox(
            "选择防御报告",
            defense_reports,
            format_func=lambda p: p.name
        )
        report = load_json(selected)
        if report and "results" in report:
            results = report["results"]

            # 提取关键指标
            if "cleansing_comparison" in results:
                comp = results["cleansing_comparison"]
                c1, c2, c3 = st.columns(3)
                with c1:
                    orig_ca = comp["original_model"]["clean_accuracy"]
                    def_ca = comp["defended_model"]["clean_accuracy"]
                    metric_card("Clean Accuracy", f"{def_ca*100:.2f}%",
                              delta=f"{'↑' if def_ca > orig_ca else '↓'}{abs(def_ca - orig_ca)*100:.1f}%")
                with c2:
                    orig_asr = comp["original_model"]["attack_success_rate"]
                    def_asr = comp["defended_model"]["attack_success_rate"]
                    metric_card("Attack Success Rate", f"{def_asr*100:.2f}%",
                              delta=f"{'↓'}{abs(def_asr - orig_asr)*100:.1f}%", delta_color="inverse")
                with c3:
                    reduction = comp["improvements"]["asr_reduction"]
                    metric_card("ASR 降幅", f"{reduction*100:.1f}%")

            if "finetuning_comparison" in results:
                comp2 = results["finetuning_comparison"]
                c1, c2, c3 = st.columns(3)
                with c1:
                    orig_ca = comp2["original_model"]["clean_accuracy"]
                    def_ca = comp2["defended_model"]["clean_accuracy"]
                    metric_card("微调 CA", f"{def_ca*100:.2f}%",
                              delta=f"{'↑' if def_ca > orig_ca else '↓'}{abs(def_ca - orig_ca)*100:.1f}%")
                with c2:
                    orig_asr = comp2["original_model"]["attack_success_rate"]
                    def_asr = comp2["defended_model"]["attack_success_rate"]
                    metric_card("微调 ASR", f"{def_asr*100:.2f}%",
                              delta=f"{'↓'}{abs(def_asr - orig_asr)*100:.1f}%", delta_color="inverse")
                with c3:
                    reduction = comp2["improvements"]["asr_reduction"]
                    metric_card("微调 ASR 降幅", f"{reduction*100:.1f}%")


# ============================================================
# Tab 4: 实时预测
# ============================================================
with tabs[4]:
    st.subheader("交互式模型预测演示")
    st.caption("选择一个 MNIST 测试样本，观察干净模型和后门模型的行为差异")

    device = get_device()

    # 交互演示固定对比干净模型与 10% 投毒后门模型。
    clean_path = MODEL_DIR / "clean_mnist_simplecnn.pth"
    bd_path = MODEL_DIR / "backdoor_mnist_simplecnn_poison0.1.pth"

    if clean_path.exists() and bd_path.exists():
        clean_model = load_model(clean_path)
        bd_model = load_model(bd_path)

        # 加载原始干净测试集；触发图片仅在内存中临时构造。
        from src.dataset import load_mnist_splits
        _, test_ds = load_mnist_splits(ROOT / "data" / "MNIST" / "raw", download=False)
        from src.trigger import add_trigger

        # 滑块只选择索引，保证四次预测始终作用于同一个原始样本。
        sample_idx = st.slider("选择测试样本", 0, len(test_ds) - 1, 42)

        image, true_label = test_ds[sample_idx]
        triggered_image = add_trigger(image.clone(), trigger_size=3, position="bottom_right")

        # 对同一图像执行四次推理，隔离模型类型与触发器两个变量。
        with torch.no_grad():
            img_batch = image.unsqueeze(0)
            trig_batch = triggered_image.unsqueeze(0)

            # 干净模型：原图与加触发图各预测一次。
            clean_out = torch.softmax(clean_model(img_batch), dim=1)[0]
            clean_pred = clean_out.argmax().item()
            clean_conf = clean_out.max().item()
            clean_target_conf = clean_out[0].item()  # target class 0

            clean_trig_out = torch.softmax(clean_model(trig_batch), dim=1)[0]
            clean_trig_pred = clean_trig_out.argmax().item()
            clean_trig_conf = clean_trig_out.max().item()
            clean_trig_target_conf = clean_trig_out[0].item()

            # 后门模型：同样分别预测，观察目标类 0 的概率是否突增。
            bd_out = torch.softmax(bd_model(img_batch), dim=1)[0]
            bd_pred = bd_out.argmax().item()
            bd_conf = bd_out.max().item()
            bd_target_conf = bd_out[0].item()

            bd_trig_out = torch.softmax(bd_model(trig_batch), dim=1)[0]
            bd_trig_pred = bd_trig_out.argmax().item()
            bd_trig_conf = bd_trig_out.max().item()
            bd_trig_target_conf = bd_trig_out[0].item()

        # 先渲染视觉对比，再展示结构化预测表和十类概率柱状图。
        col_img, col_pred = st.columns([1, 2])

        with col_img:
            import matplotlib.pyplot as plt
            import io

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
            ax1.imshow(image.squeeze(), cmap="gray")
            ax1.set_title(f"Original (Label: {true_label})")
            ax1.axis("off")
            ax2.imshow(triggered_image.squeeze(), cmap="gray")
            ax2.set_title("With Trigger")
            ax2.axis("off")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            st.image(buf, caption=f"样本 #{sample_idx}")

        with col_pred:
            # 四行表格确保用户可直接比较同一输入在不同模型下的输出。
            pred_data = {
                "模型": ["干净模型", "干净模型+触发", "后门模型(10%)", "后门模型(10%)+触发"],
                "输入": ["原始图像", "+触发器", "原始图像", "+触发器"],
                "预测": [str(clean_pred), str(clean_trig_pred), str(bd_pred), str(bd_trig_pred)],
                "置信度": [f"{clean_conf:.4f}", f"{clean_trig_conf:.4f}",
                         f"{bd_conf:.4f}", f"{bd_trig_conf:.4f}"],
                "目标类(0)置信度": [f"{clean_target_conf:.4f}", f"{clean_trig_target_conf:.4f}",
                            f"{bd_target_conf:.4f}", f"{bd_trig_target_conf:.4f}"],
            }
            st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)

            # 仅当后门模型从非 0 原预测转为目标类 0 时提示明显攻击成功。
            st.divider()
            if bd_trig_pred == 0 and bd_pred != 0:
                st.error(f"🚨 **后门攻击成功!** 触发图像被强制预测为目标类 0 (原始预测: {bd_pred})")
            elif clean_trig_pred != 0 and bd_trig_pred == 0:
                st.success(f"⚠️ 干净模型免疫触发, 后门模型被操控: {clean_trig_pred} → {bd_trig_pred}")
            else:
                st.info("当前样本未显示明显后门效果，尝试选择其他样本")

        # 四个子图逐类展示 softmax 概率，不只显示 argmax 结果。
        st.divider()
        st.subheader("置信度分布对比")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        titles = [
            "Clean Model - Original Image",
            "Clean Model - Triggered Image",
            "Backdoor Model - Original Image",
            "Backdoor Model - Triggered Image",
        ]
        all_probs = [
            clean_out.numpy(), clean_trig_out.numpy(),
            bd_out.numpy(), bd_trig_out.numpy(),
        ]

        for ax, title, probs in zip(axes.flat, titles, all_probs):
            colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(10)]
            ax.bar(range(10), probs, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Class")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(10))
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        st.image(buf, caption="预测置信度分布", use_container_width=True)

    else:
        st.warning("模型未找到。请先运行 `python main.py run-all` 训练模型。")


# ============================================================
# Tab 5: 原始报告
# ============================================================
with tabs[5]:
    st.subheader("原始 JSON 报告")
    reports = sorted(REPORT_DIR.glob("*.json"))
    if reports:
        selected = st.selectbox(
            "选择报告文件",
            reports,
            format_func=lambda p: p.name
        )
        st.json(load_json(selected))
    else:
        st.warning("当前没有 JSON 报告。")

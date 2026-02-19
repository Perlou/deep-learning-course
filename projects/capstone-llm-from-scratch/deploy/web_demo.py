"""
web_demo.py — Gradio Web 演示界面
====================================

提供浏览器可视化对话界面，支持:
  - 多轮对话
  - 参数调节面板
  - 模型信息展示
  - 公网分享链接

启动方式:
  python deploy/web_demo.py --model outputs/dpo/final.pth
  python deploy/web_demo.py --model outputs/sft/final.pth --share  # 创建公网链接
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import gradio as gr

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.inference.generate import generate_text
from src.training.trainer_utils import get_device, load_checkpoint


# ============================================================
# 全局模型
# ============================================================

model = None
tokenizer = None
device = None
param_info_str = ""


def load_model_global(config_path: str, model_path: str, tokenizer_path: str):
    """加载模型"""
    global model, tokenizer, device, param_info_str

    device = get_device()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    tokenizer = ClearMindTokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    p = model.count_parameters()
    param_info_str = (
        f"**ClearMind** | "
        f"参数量: {p['total_millions']:.1f}M | "
        f"d_model: {model_config.d_model} | "
        f"层数: {model_config.n_layers} | "
        f"设备: {device} | "
        f"模型: {os.path.basename(model_path)}"
    )

    print(f"✅ 模型加载完成: {p['total_millions']:.1f}M params on {device}")


# ============================================================
# 对话逻辑
# ============================================================


def chat_fn(
    message: str,
    history: list,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
):
    """处理对话请求

    Args:
        message:     用户输入
        history:     对话历史 [(user, bot), ...]
        temperature: 温度
        top_k:       Top-k
        top_p:       Top-p
        max_tokens:  最大生成长度
    """
    if model is None:
        return "❌ 模型未加载，请检查启动参数"

    # 构建对话 prompt (包含历史)
    prompt_parts = []
    for user_msg, bot_msg in history:
        prompt_parts.append(f"Human: {user_msg}")
        if bot_msg:
            prompt_parts.append(f"Assistant: {bot_msg}")
    prompt_parts.append(f"Human: {message}")
    prompt_parts.append("Assistant: ")

    prompt = "\n".join(prompt_parts)

    # 生成回复
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )

    # 提取 Assistant 回复
    if "Assistant: " in output:
        reply = output.split("Assistant: ")[-1]
    else:
        reply = output

    reply = reply.replace("</s>", "").replace("<s>", "").strip()

    # 如果回复包含 "Human:" 则截断
    if "Human:" in reply:
        reply = reply.split("Human:")[0].strip()

    return reply


# ============================================================
# 构建 Gradio 界面
# ============================================================


def create_demo():
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="ClearMind 对话",
        theme=gr.themes.Soft(),
    ) as demo:
        # 标题
        gr.Markdown(
            """
            # 🧠 ClearMind — 从零训练的大语言模型
            > 纯 PyTorch 手写 | RoPE + RMSNorm + SwiGLU + GQA | Pretrain → SFT → DPO
            """
        )

        # 模型信息
        gr.Markdown(param_info_str)

        with gr.Row():
            # 左侧: 对话区域
            with gr.Column(scale=4):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[
                        gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="越高越随机，越低越确定",
                        ),
                        gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=50,
                            step=1,
                            label="Top-K",
                            info="候选词数量",
                        ),
                        gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-P",
                            info="核采样阈值",
                        ),
                        gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=300,
                            step=50,
                            label="最大生成长度",
                            info="最多生成的 token 数",
                        ),
                    ],
                    additional_inputs_accordion=gr.Accordion("⚙️ 生成参数", open=False),
                    examples=[
                        ["你好，请做一个自我介绍"],
                        ["请解释什么是深度学习"],
                        ["写一首关于春天的诗"],
                        ["列出三个学习编程的建议"],
                        ["什么是 Transformer 架构？"],
                    ],
                    retry_btn="🔄 重新生成",
                    undo_btn="↩️ 撤销",
                    clear_btn="🗑️ 清空对话",
                )

        # 底部说明
        gr.Markdown(
            """
            ---
            > 💡 **提示**: ClearMind 是一个教育项目的小型模型，回复质量有限。
            > 模型仅用于学习和演示目的。
            """
        )

    return demo


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="ClearMind Web 演示")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/dpo/final.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="创建公网分享链接")
    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.tokenizer):
        print(f"❌ 分词器不存在: {args.tokenizer}")
        sys.exit(1)

    if not os.path.exists(args.model):
        for path in [
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("❌ 未找到任何 checkpoint，请先完成训练")
            sys.exit(1)

    # 加载模型
    load_model_global(args.config, args.model, args.tokenizer)

    # 创建并启动 Gradio
    demo = create_demo()

    print(f"\n🚀 启动 Web 界面: http://localhost:{args.port}")
    if args.share:
        print("   正在创建公网分享链接...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

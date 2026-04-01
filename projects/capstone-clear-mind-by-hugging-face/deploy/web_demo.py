"""
web_demo.py — Gradio Web 演示界面 (HF 版)
============================================

使用 pipeline + Gradio 提供浏览器可视化对话界面。

from-scratch 对比:
  - from-scratch: GPT + load_checkpoint + 手写 generate_text + Gradio
  - HF 版: from_pretrained + pipeline / model.generate() + Gradio

启动方式:
  python deploy/web_demo.py --model outputs/sft
  python deploy/web_demo.py --model outputs/dpo --share
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gradio as gr

from model import ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from inference.generate import generate_text, extract_reply


# ============================================================
# 全局模型
# ============================================================

_model = None
_tokenizer = None
_param_info = ""


def load_model_global(model_path: str, tokenizer_path: str):
    """加载模型

    from-scratch 对比:
      - from-scratch: GPT(config) + load_checkpoint() 两步
      - HF 版: from_pretrained() 一步
    """
    global _model, _tokenizer, _param_info

    print(f"加载模型: {model_path}")
    _model = ClearMindForCausalLM.from_pretrained(model_path)
    _model.eval()

    print(f"加载 tokenizer: {tokenizer_path}")
    _tokenizer = ClearMindTokenizer.load(tokenizer_path)

    param_count = sum(p.numel() for p in _model.parameters())
    config = _model.config
    _param_info = (
        f"**ClearMind (HF)** | "
        f"参数量: {param_count / 1e6:.1f}M | "
        f"hidden_size: {config.hidden_size} | "
        f"层数: {config.num_hidden_layers} | "
        f"模型: {os.path.basename(model_path)}"
    )

    print(f"模型加载完成: {param_count / 1e6:.1f}M params")


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
    """处理对话请求"""
    if _model is None:
        return "模型未加载，请检查启动参数"

    # 构建对话 prompt (包含历史)
    prompt_parts = []
    for user_msg, bot_msg in history:
        prompt_parts.append(f"Human: {user_msg}")
        if bot_msg:
            prompt_parts.append(f"Assistant: {bot_msg}")
    prompt_parts.append(f"Human: {message}")
    prompt_parts.append("Assistant: ")
    prompt = "\n".join(prompt_parts)

    # 生成回复 (model.generate 一行完成)
    output = generate_text(
        model=_model,
        tokenizer=_tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    reply = extract_reply(output, prompt)
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
        gr.Markdown(
            """
            # ClearMind — HuggingFace 生态大语言模型
            > RoPE + RMSNorm + SwiGLU + GQA | Pretrain → SFT → DPO | PEFT LoRA
            """
        )

        gr.Markdown(_param_info)

        chatbot = gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[
                gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature",
                          info="越高越随机，越低越确定"),
                gr.Slider(1, 200, value=50, step=1, label="Top-K",
                          info="候选词数量"),
                gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-P",
                          info="核采样阈值"),
                gr.Slider(50, 500, value=300, step=50, label="最大生成长度",
                          info="最多生成的 token 数"),
            ],
            additional_inputs_accordion=gr.Accordion("生成参数", open=False),
            examples=[
                ["你好，请做一个自我介绍"],
                ["请解释什么是深度学习"],
                ["写一首关于春天的诗"],
                ["列出三个学习编程的建议"],
                ["什么是 Transformer 架构？"],
            ],
            retry_btn="重新生成",
            undo_btn="撤销",
            clear_btn="清空对话",
        )

        gr.Markdown(
            """
            ---
            > ClearMind 是一个教育项目的小型模型，回复质量有限，仅用于学习和演示目的。
            """
        )

    return demo


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="ClearMind Web 演示 (HF 版)")
    parser.add_argument("--model", type=str, help="HF 格式模型目录")
    parser.add_argument("--tokenizer", type=str, default="outputs/tokenizer")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="创建公网分享链接")
    args = parser.parse_args()

    # 自动查找模型
    if not args.model:
        for path in ["outputs/dpo", "outputs/sft", "outputs/pretrain"]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("未找到任何模型目录，请先完成训练")
            sys.exit(1)

    load_model_global(args.model, args.tokenizer)

    demo = create_demo()

    print(f"\n启动 Web 界面: http://localhost:{args.port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

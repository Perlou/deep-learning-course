"""
web_demo.py — Gradio Web 演示界面
==================================

基于 minimind chat_template 的可视化对话界面，支持：
  - 多轮对话历史
  - System prompt 注入
  - open_thinking 自适应思考开关
  - 采样参数实时调节
  - Stop 中断生成

用法:
  python deploy/web_demo.py --config configs/main.yaml --model outputs/dpo/final.pth
  python deploy/web_demo.py --port 7860 --share   # 公网分享链接
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="ClearMind Gradio Web Demo")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    # 自动找模型
    model_path = args.model
    if model_path is None:
        for cand in ("outputs/dpo/final.pth", "outputs/sft/final.pth"):
            if os.path.exists(cand):
                model_path = cand
                break
    if model_path is None or not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        sys.exit(1)

    # ===== 模型加载 =====
    import yaml
    import torch

    from src.model.config import ModelConfig
    from src.model.gpt import GPT
    from src.training.trainer_utils import get_device, load_checkpoint
    from src.inference.generate import generate_chat
    from scripts.train import load_tokenizer

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = ModelConfig(**config["model"])
    tokenizer = load_tokenizer(config, args.tokenizer)
    if tokenizer.vocab_size != model_config.vocab_size:
        model_config.vocab_size = tokenizer.vocab_size

    device = get_device()
    model = GPT(model_config).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()

    release = config.get("release", {})
    model_name = release.get("display_name") or release.get("model_name") or "ClearMind"

    # ===== Gradio =====
    try:
        import gradio as gr
    except ImportError:
        print("❌ 请先安装: pip install gradio")
        sys.exit(1)

    def chat_fn(
        message: str,
        history: list[dict],
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        open_thinking: bool,
    ):
        """Gradio chatbot 回调

        history 是 messages 风格列表 [{role, content}, ...]（Gradio Chatbot type='messages'）
        """
        # 组装 messages
        messages: list[dict] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        # 历史消息透传
        for h in history:
            if isinstance(h, dict):
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": message})

        # 生成
        reply = generate_chat(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=50,
            device=device,
            open_thinking=open_thinking,
        )

        # 处理 thinking
        if open_thinking and "</think>" in reply:
            think, _, ans = reply.partition("</think>")
            think = think.replace("<think>", "").strip()
            display = (
                f"<details><summary>💭 思考过程</summary>\n\n{think}\n\n</details>\n\n{ans.strip()}"
                if think else ans.strip()
            )
        else:
            display = reply
        return display

    with gr.Blocks(title=f"{model_name} - 对话演示") as demo:
        gr.Markdown(f"# 🧠 {model_name}\n\n基于 minimind 数据训练的中文小模型，发布于 HuggingFace / ModelScope")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=560,
                    show_copy_button=True,
                    avatar_images=(None, "🧠"),
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送...",
                        show_label=False,
                        scale=8,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### ⚙️ 生成参数")
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value="你是一个有用的 AI 助手。",
                    lines=2,
                )
                open_thinking = gr.Checkbox(label="自适应思考 (open_thinking)", value=False)
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                max_new = gr.Slider(32, 1024, value=512, step=32, label="Max new tokens")
                gr.Markdown(f"**Device**: {device}")
                gr.Markdown(f"**Model**: `{Path(model_path).name}`")

        # ----- 事件 -----
        def respond(message, chat_history, system_prompt, temperature, top_p, max_new, open_thinking):
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            yield "", chat_history  # 立即清空输入框
            reply = chat_fn(
                message, chat_history[:-1], system_prompt, temperature, top_p, max_new, open_thinking
            )
            chat_history.append({"role": "assistant", "content": reply})
            yield "", chat_history

        send_btn.click(
            respond,
            inputs=[user_input, chatbot, system_prompt, temperature, top_p, max_new, open_thinking],
            outputs=[user_input, chatbot],
        )
        user_input.submit(
            respond,
            inputs=[user_input, chatbot, system_prompt, temperature, top_p, max_new, open_thinking],
            outputs=[user_input, chatbot],
        )
        clear_btn.click(lambda: [], None, chatbot)

    print(f"\n🚀 启动 Web Demo 于 http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

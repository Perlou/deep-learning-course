"""
chat.py — 交互式对话引擎
=========================

提供终端下的多轮对话界面。

对话模板:
  <s>Human: {用户输入}
  Assistant: {模型回复}</s>
"""

import torch
from typing import Optional

from .generate import generate_text


def chat_loop(
    model,
    tokenizer,
    device: torch.device = None,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """交互式对话循环

    Args:
        model:     GPT 模型
        tokenizer: 分词器
        device:    设备
        其他参数:  生成参数
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    print("\n" + "=" * 60)
    print("🤖 ClearMind 对话系统")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("输入 'params' 查看/修改生成参数")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 Human: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("👋 再见!")
            break

        if user_input.lower() == "clear":
            print("🗑️  对话历史已清空")
            continue

        if user_input.lower() == "params":
            print(f"  temperature: {temperature}")
            print(f"  top_k:       {top_k}")
            print(f"  top_p:       {top_p}")
            print(f"  max_tokens:  {max_new_tokens}")
            try:
                param = input("  修改参数 (格式: key=value, 回车跳过): ").strip()
                if param:
                    key, value = param.split("=")
                    key = key.strip()
                    if key == "temperature":
                        temperature = float(value)
                    elif key == "top_k":
                        top_k = int(value)
                    elif key == "top_p":
                        top_p = float(value)
                    elif key == "max_tokens":
                        max_new_tokens = int(value)
                    print(f"  ✅ {key} = {value}")
            except Exception:
                pass
            continue

        # 构建对话 prompt
        prompt = f"Human: {user_input}\nAssistant: "

        # 生成回复
        print("🤖 Assistant: ", end="", flush=True)

        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

        # 提取 Assistant 回复部分
        if "Assistant: " in response:
            reply = response.split("Assistant: ", 1)[-1]
        else:
            reply = response

        # 清理末尾的特殊 token
        reply = reply.replace("</s>", "").replace("<s>", "").strip()
        print(reply)

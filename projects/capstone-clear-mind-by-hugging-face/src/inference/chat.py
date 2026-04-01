"""
chat.py — 交互式对话引擎 (HF 版)
===================================

提供终端下的多轮对话界面。

from-scratch 对比:
  - from-scratch: chat_loop + generate_text 手写生成
  - HF 版: chat_loop + model.generate() / pipeline

对话模板:
  Human: {用户输入}
  Assistant: {模型回复}
"""

import torch

from .generate import generate_text, extract_reply


def chat_loop(
    model,
    tokenizer,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    max_history: int = 5,
):
    """交互式对话循环

    from-scratch 对比:
      - from-scratch: 手写 _build_prompt + generate_text
      - HF 版: 同样的对话管理，但生成用 model.generate()

    Args:
        model:       ClearMindForCausalLM
        tokenizer:   tokenizer
        max_history: 最大保留对话轮数
        其他参数:    生成参数
    """
    model.eval()
    history = []  # [(user_msg, assistant_reply), ...]

    print("\n" + "=" * 60)
    print("  ClearMind 对话系统 (HF 版)")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("输入 'params' 查看/修改生成参数")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nHuman: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("  对话历史已清空")
            continue

        if user_input.lower() == "params":
            print(f"  temperature:  {temperature}")
            print(f"  top_k:        {top_k}")
            print(f"  top_p:        {top_p}")
            print(f"  max_tokens:   {max_new_tokens}")
            print(f"  max_history:  {max_history}")
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
                    elif key == "max_history":
                        max_history = int(value)
                    print(f"  {key} = {value}")
            except Exception as e:
                print(f"  参数修改失败: {e}")
            continue

        # 构建多轮对话 prompt
        prompt = _build_prompt(history, user_input)

        # 生成回复 (model.generate 一行完成)
        print("Assistant: ", end="", flush=True)

        response = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        reply = extract_reply(response, prompt)
        print(reply)

        # 保存对话历史
        history.append((user_input, reply))
        if len(history) > max_history:
            history = history[-max_history:]


def _build_prompt(history: list[tuple[str, str]], current_input: str) -> str:
    """将对话历史和当前输入拼接为完整 prompt"""
    prompt = ""
    for user_msg, reply in history:
        prompt += f"Human: {user_msg}\nAssistant: {reply}\n"
    prompt += f"Human: {current_input}\nAssistant: "
    return prompt

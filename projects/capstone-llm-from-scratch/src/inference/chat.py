"""
chat.py — 交互式对话引擎
=========================

提供终端下的多轮对话界面，使用 :meth:`HFTokenizer.apply_chat_template` 渲染
完整对话历史（含可选 system prompt 与 ``open_thinking`` 自适应思考）。

与旧实现（``<s>Human:/Assistant:`` 字符串拼接）相比：
  - 完全对齐 minimind / Qwen3 的 ``<|im_start|>role\\n...<|im_end|>\\n`` 格式
  - 自动支持 ``open_thinking=True`` 让模型在 ``<think>...</think>`` 中先推理
  - 多轮对话历史不会因为字符串截断错位
"""

from __future__ import annotations

import torch

from .generate import generate


def _slash_command(
    cmd: str,
    history: list[dict],
    state: dict,
) -> bool:
    """处理斜杠命令；返回 True 表示已处理（应跳过对话生成）"""
    cmd = cmd.lower()
    if cmd in ("quit", "exit", "/quit", "/exit"):
        print("👋 再见!")
        state["should_exit"] = True
        return True
    if cmd in ("clear", "/clear"):
        history.clear()
        print("🗑️  对话历史已清空")
        return True
    if cmd in ("params", "/params"):
        print(f"  temperature:    {state['temperature']}")
        print(f"  top_k:          {state['top_k']}")
        print(f"  top_p:          {state['top_p']}")
        print(f"  max_new_tokens: {state['max_new_tokens']}")
        print(f"  max_history:    {state['max_history']}")
        print(f"  open_thinking:  {state['open_thinking']}")
        try:
            param = input("  修改参数 (key=value, 回车跳过): ").strip()
            if param:
                key, value = param.split("=", 1)
                key, value = key.strip(), value.strip()
                if key in ("temperature", "top_p"):
                    state[key] = float(value)
                elif key in ("top_k", "max_new_tokens", "max_history"):
                    state[key] = int(value)
                elif key == "open_thinking":
                    state[key] = value.lower() in ("1", "true", "yes", "y")
                else:
                    print(f"  ⚠️ 未知参数: {key}")
                    return True
                print(f"  ✅ {key} = {state[key]}")
        except Exception as e:
            print(f"  ⚠️ 参数修改失败: {e}")
        return True
    if cmd in ("system", "/system"):
        new_sys = input("  新 system prompt（回车清空）: ").strip()
        state["system_prompt"] = new_sys or None
        print(f"  ✅ system = {state['system_prompt']!r}")
        return True
    if cmd in ("think", "/think"):
        state["open_thinking"] = not state["open_thinking"]
        print(f"  ✅ open_thinking = {state['open_thinking']}")
        return True
    if cmd in ("help", "/help", "?", "/?"):
        print(
            "  命令：\n"
            "    /clear        清空对话历史\n"
            "    /params       查看/修改生成参数\n"
            "    /system       设置/清空 system prompt\n"
            "    /think        切换 open_thinking 模式\n"
            "    /help         显示此帮助\n"
            "    /quit         退出"
        )
        return True
    return False


def chat_loop(
    model,
    tokenizer,
    device: torch.device = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    max_history: int = 8,
    system_prompt: str | None = None,
    open_thinking: bool = False,
):
    """交互式多轮对话循环（使用 chat_template 渲染）

    Args:
        model:              GPT 模型（需暴露 ``forward``、``config``）
        tokenizer:          :class:`HFTokenizer` 实例（必须有 apply_chat_template）
        device:             计算设备（None 时取 model 所在设备）
        max_new_tokens:     单轮最多生成的 token 数
        temperature/top_k/top_p/repetition_penalty: 采样参数
        max_history:        保留的最大对话轮数（user+assistant 算 1 轮）
        system_prompt:      可选 system 消息，``None`` 表示不注入
        open_thinking:      渲染 prompt 时是否带 ``<think>\\n`` 引导，让模型先思考
    """
    if device is None:
        device = next(model.parameters()).device
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError(
            "chat_loop 要求 tokenizer 支持 apply_chat_template；请用 HFTokenizer。"
        )

    model.eval()
    history: list[dict] = []  # 标准 messages 列表 [{role, content}, ...]

    state = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_history": max_history,
        "open_thinking": open_thinking,
        "system_prompt": system_prompt,
        "should_exit": False,
    }

    print("\n" + "=" * 60)
    print("🤖 ClearMind 对话系统")
    print("=" * 60)
    print("输入 /help 查看命令，/quit 退出。")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见!")
            break

        if not user_input:
            continue

        # 处理斜杠命令
        if _slash_command(user_input, history, state):
            if state["should_exit"]:
                break
            continue

        # ========== 构造 messages ==========
        messages: list[dict] = []
        if state["system_prompt"]:
            messages.append({"role": "system", "content": state["system_prompt"]})
        # 拼接历史
        messages.extend(history)
        # 当前轮 user
        messages.append({"role": "user", "content": user_input})

        # ========== 渲染 prompt ==========
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            open_thinking=state["open_thinking"],
        )

        # ========== Tokenize ==========
        # apply_chat_template 已经包含必要的 BOS/特殊 token，这里 add_bos=False
        prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
        max_seq_len = getattr(getattr(model, "config", None), "max_seq_len", 1024)

        # 防御性夹紧：当 max_new_tokens >= max_seq_len 时（典型场景：tiny
        # max_seq_len=128 但用户用了默认 max_new_tokens=512），budget 会变 0/负，
        # `prompt_ids[-keep:]` 在 keep 为负时返回 [] → input_ids.shape=(1,0) →
        # model.forward 里 torch.full((0,0), ...) 在 MPS 上直接断言失败。
        max_new = state["max_new_tokens"]
        if max_new >= max_seq_len:
            new_max_new = max(1, max_seq_len // 2)
            print(
                f"  ⚠️  max_new_tokens ({max_new}) ≥ max_seq_len ({max_seq_len})，"
                f"临时降到 {new_max_new}（建议改用更大的配置）"
            )
            max_new = new_max_new
        budget = max(1, max_seq_len - max_new)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
        # 极端 case 兜底：渲染后竟然是空（不应该发生）
        if not prompt_ids:
            print("  ⚠️ prompt 为空，跳过本轮")
            continue

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # ========== 生成 ==========
        print("🤖 ClearMind: ", end="", flush=True)
        try:
            output_ids = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max_new,
                temperature=state["temperature"],
                top_k=state["top_k"],
                top_p=state["top_p"],
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_id,
            )
        except Exception as e:
            print(f"\n  ⚠️ 生成失败: {e}")
            continue

        # 只取新生成的部分
        new_ids = output_ids[0, len(prompt_ids):].tolist()
        # 截断到第一个 EOS
        if tokenizer.eos_id in new_ids:
            new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # 如果开了 open_thinking，把 <think>...</think> 单独显示
        if state["open_thinking"] and "</think>" in reply:
            think_part, _, answer_part = reply.partition("</think>")
            think_part = think_part.replace("<think>", "").strip()
            if think_part:
                print(f"\n  💭 [thinking]\n  {think_part}\n")
            print(f"  {answer_part.strip()}")
            display_reply = answer_part.strip()
        else:
            print(reply)
            display_reply = reply

        # ========== 更新历史 ==========
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": display_reply})

        # 控制历史长度（按"轮"计算，1 轮 = user+assistant 两条）
        max_msgs = state["max_history"] * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]

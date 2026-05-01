"""
generate.py — 文本生成引擎
===========================

提供灵活的文本生成接口，支持多种采样策略。

采样策略:
  1. Greedy:    每步选择概率最高的 token (确定性)
  2. Top-k:     只在概率最高的 k 个 token 中采样
  3. Top-p:     (Nucleus Sampling) 在累积概率达到 p 的最小集合中采样
  4. Temperature: 控制分布的"锐度" — 高 = 更随机, 低 = 更确定

这些策略可以组合使用:
  Temperature 先调整分布 → Top-k 裁剪 → Top-p 裁剪 → 采样
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """自回归文本生成 (使用 KV Cache 加速)

    Args:
        model:              GPT 模型
        input_ids:          输入 token [1, seq_len]
        max_new_tokens:     最多生成的 token 数
        temperature:        温度参数 (>0)
        top_k:              Top-k 采样的 k 值
        top_p:              Top-p 采样的 p 值
        repetition_penalty: 重复惩罚 (>1.0 减少重复)
        eos_token_id:       结束 token ID

    Returns:
        生成的完整序列 [1, seq_len + generated_len]
    """
    model.eval()
    device = input_ids.device

    # 获取 max_seq_len
    if hasattr(model, "config"):
        max_seq_len = model.config.max_seq_len
    else:
        max_seq_len = 512

    # ========== Prefill: 处理完整 prompt ==========
    cond_ids = input_ids[:, -max_seq_len:]
    logits, _, kv_caches = model(cond_ids, use_cache=True)
    logits = logits[:, -1, :]  # [1, vocab_size]

    for _ in range(max_new_tokens):
        # 重复惩罚: 降低已出现 token 的概率
        if repetition_penalty != 1.0:
            unique_ids = torch.unique(input_ids[0])
            penalty_logits = logits[0, unique_ids]
            logits[0, unique_ids] = torch.where(
                penalty_logits > 0,
                penalty_logits / repetition_penalty,
                penalty_logits * repetition_penalty,
            )

        # 安全检查: 替换 nan/inf 为 0
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Temperature scaling
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            # temperature=0 → greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
            # Decode 下一步
            logits, _, kv_caches = model(
                next_token, use_cache=True, kv_caches=kv_caches
            )
            logits = logits[:, -1, :]
            continue

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(
                scaled_logits, min(top_k, scaled_logits.size(-1))
            )
            min_top_k = top_k_vals[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < min_top_k,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits,
            )

        # Top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过 top_p 的 token（保留第一个超过的）
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            # scatter 回原始索引空间
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scaled_logits[indices_to_remove] = float("-inf")

        # 采样
        probs = F.softmax(scaled_logits, dim=-1)
        # 防止全 -inf 导致 nan
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.size(-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

        # ========== Decode: 只输入新 token ==========
        # 检查 cache 是否超过 max_seq_len
        if kv_caches[0][0].shape[2] >= max_seq_len:
            kv_caches = [
                (k[:, :, -max_seq_len + 1 :, :], v[:, :, -max_seq_len + 1 :, :])
                for k, v in kv_caches
            ]

        logits, _, kv_caches = model(next_token, use_cache=True, kv_caches=kv_caches)
        logits = logits[:, -1, :]

    return input_ids


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = None,
    add_bos: bool = False,
    return_only_new: bool = True,
    skip_special_tokens: bool = True,
) -> str:
    """从纯文本 prompt 生成文本（不走 chat_template）

    适合：续写预训练模型的句子、Pretrain 阶段冒烟。
    多轮对话场景请使用 :func:`chat_loop` 或自行用 ``apply_chat_template``
    渲染后再调用 :func:`generate`。

    Args:
        model:               GPT 模型
        tokenizer:           分词器（HFTokenizer 或 ClearMindTokenizer）
        prompt:              输入文本
        max_new_tokens:      最多生成的 token 数
        temperature/top_k/top_p/repetition_penalty: 采样参数
        device:              设备（None = model 所在设备）
        add_bos:             是否在 prompt 开头插入 BOS。
                             *minimind tokenizer 默认行为*：apply_chat_template 已含 BOS，
                             纯文本续写场景才设 ``True``。
        return_only_new:     只返回新生成的 tokens（不含 prompt）
        skip_special_tokens: decode 时跳过 ``<|im_start|>`` / ``<|im_end|>`` 等
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt（默认不加 BOS，避免与 chat_template 冲突）
    input_ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    prompt_len = len(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_id,
    )

    # 决定 decode 范围
    full_ids = output_ids[0].tolist()
    new_ids = full_ids[prompt_len:] if return_only_new else full_ids
    # 截断到第一个 EOS
    if tokenizer.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]

    # decode（兼容 ClearMindTokenizer / HFTokenizer 接口差异）
    try:
        text = tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        # ClearMindTokenizer.decode 不接受 skip_special_tokens
        text = tokenizer.decode(new_ids)
    return text


def generate_chat(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = None,
    open_thinking: bool = False,
    tools: list[dict] | None = None,
) -> str:
    """从 messages 生成 assistant 回复（走 chat_template）

    自动调用 ``tokenizer.apply_chat_template(messages, add_generation_prompt=True)``
    渲染对话历史，模型只生成 assistant 段。

    Args:
        messages:        ``[{"role": "system|user|assistant|tool", "content": "..."}, ...]``
        open_thinking:   是否在 prompt 末尾插入 ``<think>\\n``，引导模型先推理
        tools:           可选 OpenAI 风格工具定义（用于 tool-use）
        其它参数同 :func:`generate_text`

    Returns:
        assistant 回复字符串（已去除 ``<|im_end|>`` 等特殊 token）
    """
    if device is None:
        device = next(model.parameters()).device
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError(
            "generate_chat 要求 tokenizer 支持 apply_chat_template，请用 HFTokenizer。"
        )

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        open_thinking=open_thinking,
    )
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
        add_bos=False,                 # apply_chat_template 已含 BOS
        return_only_new=True,
        skip_special_tokens=True,
    )

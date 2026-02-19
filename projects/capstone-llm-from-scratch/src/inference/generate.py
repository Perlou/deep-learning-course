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
            for token_id in set(input_ids[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

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

            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_mask] = float("-inf")

            scaled_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # 采样
        probs = F.softmax(scaled_logits, dim=-1)
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
    device: torch.device = None,
) -> str:
    """从文本 prompt 生成文本回复

    Args:
        model:     GPT 模型
        tokenizer: 分词器
        prompt:    输入文本
        其他参数同 generate()

    Returns:
        生成的文本
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_id,
    )

    # Decode (只取新生成的部分)
    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids)

    return text

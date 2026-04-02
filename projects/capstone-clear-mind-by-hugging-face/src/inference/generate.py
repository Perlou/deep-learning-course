"""
generate.py — 文本生成引擎 (HF 版)
====================================

使用 model.generate() + pipeline 实现文本生成。

from-scratch 对比:
  - from-scratch: 手写 ~100 行采样循环 (Temperature/Top-k/Top-p/KV Cache)
  - HF 版: model.generate() 一行完成，pipeline() 更简洁

model.generate() 的优势:
  - 内置 Greedy, Beam Search, Top-k, Top-p, Temperature 等所有采样策略
  - 内置 KV Cache 管理 (通过 prepare_inputs_for_generation)
  - 内置 repetition_penalty, no_repeat_ngram_size 等
  - 可直接接入 HF 生态 (pipeline, TextStreamer 等)

使用方法:
  from inference.generate import generate_text, create_pipeline
  text = generate_text(model, tokenizer, "你好")
  pipe = create_pipeline(model, tokenizer)
  result = pipe("你好")
"""

import torch
from transformers import pipeline, TextStreamer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """从文本 prompt 生成文本回复

    from-scratch 对比:
      - from-scratch: generate_text() 调用手写 generate() (~100行采样循环)
      - HF 版: model.generate() 一行完成

    Args:
        model:              ClearMindForCausalLM
        tokenizer:          tokenizer
        prompt:             输入文本
        max_new_tokens:     最多生成 token 数
        temperature:        温度参数
        top_k:              Top-k 采样的 k 值
        top_p:              Top-p 采样的 p 值
        repetition_penalty: 重复惩罚
        do_sample:          是否采样 (False = greedy)

    Returns:
        生成的完整文本 (含 prompt)
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 确保 prompt + 生成长度不超过 max_position_embeddings
    max_pos = getattr(model.config, "max_position_embeddings", 512)
    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = min(max_new_tokens, max(1, max_pos - prompt_len))

    # model.generate(): 一行替代 from-scratch ~100 行手写循环
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text


def extract_reply(full_text: str, prompt: str = "") -> str:
    """从生成的完整文本中提取 Assistant 回复部分

    Args:
        full_text: model.generate() 的完整输出
        prompt:    原始 prompt (用于移除前缀)

    Returns:
        Assistant 的回复文本
    """
    # 尝试提取 Assistant 回复
    if "Assistant: " in full_text:
        reply = full_text.rsplit("Assistant: ", 1)[-1]
    elif prompt and full_text.startswith(prompt):
        reply = full_text[len(prompt):]
    else:
        reply = full_text

    # 截断到第一个 "Human:" (防止模型继续生成下一轮)
    if "Human:" in reply:
        reply = reply.split("Human:")[0]

    return reply.strip()


def create_pipeline(
    model,
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """创建 HF pipeline 推理封装

    from-scratch 对比:
      - from-scratch: 无此功能，必须手动 encode → generate → decode
      - HF 版: pipeline("text-generation") 一行封装

    用法:
      pipe = create_pipeline(model, tokenizer)
      result = pipe("你好，请介绍一下自己")[0]["generated_text"]

    Args:
        model:     ClearMindForCausalLM
        tokenizer: tokenizer
        其他参数:  生成参数

    Returns:
        transformers.Pipeline 实例
    """
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> None:
    """流式生成 (逐 token 打印到终端)

    from-scratch 对比:
      - from-scratch: 手写 for 循环中逐 token print
      - HF 版: TextStreamer + model.generate()

    Args:
        model, tokenizer, prompt, 其他参数同 generate_text
    """
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # TextStreamer: 实时输出每个生成的 token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )

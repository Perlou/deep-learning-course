"""
data_utils.py — 数据处理工具函数 (HF datasets 版)
==================================================

使用 HuggingFace datasets 库加载和处理三种训练数据，
替代 from-scratch 版的手写 Dataset 类。

核心区别:
  - from-scratch: 手写 PretrainDataset/SFTDataset/DPODataset，
    在 __getitem__ 中 tokenize，手动 padding
  - HF 版: datasets.load_dataset → dataset.map(tokenize_fn) →
    DataCollator 动态处理

数据格式 (由 scripts/prepare_data.py 生成):
  - Pretrain: {"text": "..."} → data/pretrain/pretrain_data.jsonl
  - SFT: {"instruction": "...", "input": "...", "output": "..."} → data/sft/sft_data.jsonl
  - DPO: {"prompt": "...", "chosen": "...", "rejected": "..."} → data/dpo/dpo_data.jsonl
"""

from datasets import load_dataset


def load_pretrain_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """加载并处理预训练数据

    from-scratch 对比:
      - from-scratch: PretrainDataset 在 __getitem__ 中 tokenize，
        拼接所有 token 后切分为固定长度 chunk
      - HF 版: dataset.map(tokenize_fn, batched=True) 批量 tokenize，
        搭配 DataCollatorForLanguageModeling 动态生成 labels

    Args:
        data_path:  JSONL 文件路径，每行 {"text": "..."}
        tokenizer:  PreTrainedTokenizerFast 实例
        max_length: 最大序列长度
        val_ratio:  验证集比例
        seed:       随机种子

    Returns:
        dict: {"train": Dataset, "validation": Dataset}
        每条数据包含 input_ids 和 attention_mask
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # DataCollator 会动态 padding
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    split = tokenized.train_test_split(test_size=val_ratio, seed=seed)

    return {
        "train": split["train"],
        "validation": split["test"],
    }


def load_sft_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """加载并处理 SFT 指令微调数据

    from-scratch 对比:
      - from-scratch: SFTDataset 手动拼接 "Human: ... Assistant: ..." 文本，
        手动计算 prompt_len 设置 labels[:prompt_len] = -100
      - HF 版: apply_chat_template 格式化对话，tokenize 后构建 labels，
        prompt 部分设为 -100，只对 assistant 回复计算 loss

    Args:
        data_path:  JSONL 文件路径，Alpaca 格式
        tokenizer:  PreTrainedTokenizerFast 实例 (需要有 chat_template)
        max_length: 最大序列长度
        val_ratio:  验证集比例
        seed:       随机种子

    Returns:
        dict: {"train": Dataset, "validation": Dataset}
        每条数据包含 input_ids, attention_mask, labels
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_and_tokenize(examples):
        """格式化对话并 tokenize，构建 labels mask"""
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            inp = examples.get("input", [""] * len(examples["instruction"]))[i] or ""
            output = examples["output"][i]

            # 构建 user content — 与 from-scratch 版一致
            if inp:
                user_content = f"{instruction}\n{inp}"
            else:
                user_content = instruction

            # 用 chat_template 格式化 (HF 方式)
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False
            )["input_ids"]
            full_encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )

            input_ids = full_encoding["input_ids"]
            attention_mask = full_encoding["attention_mask"]

            # 构建 labels: prompt 部分设为 -100
            labels = input_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    tokenized = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    split = tokenized.train_test_split(test_size=val_ratio, seed=seed)

    return {
        "train": split["train"],
        "validation": split["test"],
    }


def load_dpo_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """加载并处理 DPO 偏好对齐数据

    from-scratch 对比:
      - from-scratch: DPODataset 手动 tokenize chosen/rejected，
        构建 4 组 tensor (chosen_input_ids, chosen_labels, rejected_*)
      - HF 版: 只需格式化为 {"prompt", "chosen", "rejected"} 字符串，
        DPOTrainer 内部自动处理 tokenize 和 loss 计算

    Args:
        data_path:  JSONL 文件路径，{"prompt", "chosen", "rejected"}
        tokenizer:  PreTrainedTokenizerFast 实例 (需要有 chat_template)
        max_length: 最大序列长度 (DPO 中用于格式化参考)
        val_ratio:  验证集比例
        seed:       随机种子

    Returns:
        dict: {"train": Dataset, "validation": Dataset}
        每条数据包含 prompt, chosen, rejected (字符串格式)
    """
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_dpo(examples):
        """格式化为 DPOTrainer 需要的字符串格式"""
        prompts = []
        chosens = []
        rejecteds = []

        for i in range(len(examples["prompt"])):
            raw_prompt = examples["prompt"][i]
            chosen = examples["chosen"][i]
            rejected = examples["rejected"][i]

            # 用 chat_template 格式化 prompt
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(formatted_prompt)
            chosens.append(chosen)
            rejecteds.append(rejected)

        return {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }

    formatted = dataset.map(
        format_dpo,
        batched=True,
        remove_columns=dataset.column_names,
    )

    split = formatted.train_test_split(test_size=val_ratio, seed=seed)

    return {
        "train": split["train"],
        "validation": split["test"],
    }

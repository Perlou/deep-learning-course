"""
sft.py — SFT 指令微调核心逻辑 (HF Trainer 版)
================================================

使用 HuggingFace Trainer 执行 SFT (Supervised Fine-Tuning)，
让预训练模型从"续写文本"变成"按指令回答问题"。

from-scratch 对比:
  - SFTTrainer(BaseTrainer) 手写 epoch loop → HF Trainer.train()
  - 手动 loss mask (labels[:prompt_len] = -100) → load_sft_dataset 已处理
  - 手动加载预训练 checkpoint → model = from_pretrained(pretrained_path)
  - apply_lora + rebuild_optimizer → PEFT 集成 (阶段 8)

SFT 与预训练的关键区别:
  1. 只在 Assistant 回复部分计算 Loss (labels mask)
  2. 使用更小的学习率 (避免灾难性遗忘)
  3. 按 epoch 训练 (不用 max_steps)
  4. 可从预训练 checkpoint 加载权重
"""

import yaml
import torch
from transformers import (
    Trainer,
    TrainingArguments,
)

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from data.data_utils import load_sft_dataset
from training.callbacks import ClearMindLoggingCallback


class SFTDataCollator:
    """SFT 数据 collator — 动态 padding input_ids, attention_mask, labels

    from-scratch 对比:
      - from-scratch: SFTDataset.__getitem__ 中固定长度 padding
      - HF 版: DataCollator 在 batch 时动态 padding 到 batch 内最长序列

    labels 使用 -100 padding (CrossEntropyLoss 忽略),
    input_ids 使用 pad_token_id padding。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_sft_args(sft_config: dict, output_dir: str) -> TrainingArguments:
    """从 YAML sft 节构建 TrainingArguments

    SFT 使用 epoch-based 训练（不同于 pretrain 的 max_steps）。
    tiny.yaml sft 节字段: per_device_train_batch_size, gradient_accumulation_steps,
    num_train_epochs, learning_rate, weight_decay, logging_steps, bf16, report_to

    Args:
        sft_config: YAML 中 sft 节的字典
        output_dir: 输出目录

    Returns:
        TrainingArguments 实例
    """
    args_dict = dict(sft_config)

    # SFT 合理默认值
    args_dict.setdefault("lr_scheduler_type", "cosine")
    args_dict.setdefault("warmup_ratio", 0.1)
    args_dict.setdefault("save_strategy", "epoch")
    args_dict.setdefault("eval_strategy", "epoch")
    args_dict.setdefault("dataloader_drop_last", False)
    args_dict.setdefault("remove_unused_columns", False)
    args_dict.setdefault("gradient_checkpointing", False)

    return TrainingArguments(
        output_dir=output_dir,
        **args_dict,
    )


def run_sft(
    config_path: str,
    data_path: str,
    tokenizer_path: str,
    output_dir: str,
    pretrained_path: str | None = None,
    num_train_epochs: int | None = None,
    max_steps: int | None = None,
    use_lora: bool = False,
) -> None:
    """执行 SFT 指令微调

    对应 from-scratch 版 SFTTrainer.train() 的完整流程。

    SFT 流程:
      1. 加载预训练模型 (from_pretrained 或随机初始化)
      2. 可选: 应用 LoRA (use_lora=True)
      3. 加载 SFT 数据 (load_sft_dataset, labels 已含 -100 mask)
      4. 用 HF Trainer 训练
      5. 保存微调后的模型

    Args:
        config_path:       YAML 配置文件路径
        data_path:         SFT 数据 JSONL 路径
        tokenizer_path:    tokenizer 目录路径
        output_dir:        输出目录
        pretrained_path:   预训练模型路径 (HF 格式目录, 可选)
        num_train_epochs:  覆盖 config 中的 epoch 数 (可选)
        max_steps:         覆盖为 step-based 训练 (可选, 用于测试)
        use_lora:          是否使用 LoRA 微调
    """
    # 1. 加载 YAML 配置
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    sft_config = raw_config.get("sft", {})

    # 2. 加载 tokenizer
    tokenizer = ClearMindTokenizer.load(tokenizer_path)

    # 3. 创建或加载模型
    if pretrained_path:
        print(f"从预训练模型加载: {pretrained_path}")
        model = ClearMindForCausalLM.from_pretrained(pretrained_path)
    else:
        print("未指定预训练模型, 从随机初始化开始 SFT")
        model_config = ClearMindConfig.from_yaml(config_path)
        # 同步 vocab_size
        if tokenizer.vocab_size != model_config.vocab_size:
            model_config.vocab_size = tokenizer.vocab_size
        model = ClearMindForCausalLM(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 3.5 可选: 应用 LoRA
    if use_lora:
        from training.lora import create_lora_config, apply_peft_lora
        lora_config_dict = raw_config.get("lora", {})
        lora_config = create_lora_config(lora_config_dict)
        model = apply_peft_lora(model, lora_config)

    # 4. 加载 SFT 数据 — labels 已含 -100 mask (prompt 部分)
    max_length = model.config.max_position_embeddings
    datasets = load_sft_dataset(data_path, tokenizer, max_length=max_length)
    print(f"训练集: {len(datasets['train'])} 条, 验证集: {len(datasets['validation'])} 条")

    # 5. DataCollator — SFT 数据已有 labels，需动态 padding (labels 用 -100)
    data_collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    # 6. 覆盖参数
    if num_train_epochs is not None:
        sft_config["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        sft_config["max_steps"] = max_steps
        # step-based 模式下关闭 epoch-based 的 eval/save
        sft_config["eval_strategy"] = "no"
        sft_config["save_strategy"] = "no"

    # 7. 构建 TrainingArguments 和 Trainer
    training_args = create_sft_args(sft_config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[ClearMindLoggingCallback()],
    )

    # 8. 训练
    trainer.train()

    # 9. 保存最终模型
    if use_lora:
        # 保存 LoRA adapter
        adapter_dir = output_dir + "_lora_adapter"
        trainer.model.save_pretrained(adapter_dir)
        print(f"LoRA adapter 已保存到: {adapter_dir}")
        # 合并 LoRA 并保存完整模型
        from training.lora import merge_lora_weights
        merged_model = merge_lora_weights(trainer.model)
        merged_model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSFT 模型已保存到: {output_dir}")

"""
pretrain.py — 预训练核心逻辑 (HF Trainer 版)
==============================================

使用 HuggingFace Trainer 替代 from-scratch 的手写训练循环。

from-scratch 对比:
  - BaseTrainer.__init__ (AdamW, GradScaler, scheduler) → TrainingArguments
  - PreTrainer.train() while loop → Trainer.train()
  - 手动梯度累积 / clip / log → Trainer 内部自动处理
  - save_checkpoint / load_checkpoint → Trainer.save_model / resume_from_checkpoint
  - evaluate_loss + EarlyStopping → Trainer evaluation + load_best_model_at_end
"""

import yaml
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from data.data_utils import load_pretrain_dataset
from training.callbacks import ClearMindLoggingCallback


def create_training_args(pretrain_config: dict, output_dir: str) -> TrainingArguments:
    """从 YAML pretrain 节构建 TrainingArguments

    tiny.yaml 的 pretrain 字段已使用 HF 兼容名（per_device_train_batch_size 等），
    近乎直接透传。额外设置一些合理默认值。

    Args:
        pretrain_config: YAML 中 pretrain 节的字典
        output_dir: 模型和 checkpoint 的输出目录

    Returns:
        TrainingArguments 实例
    """
    args_dict = dict(pretrain_config)
    max_steps = args_dict.get("max_steps", 200)

    # 当 max_steps 较小时（冒烟测试），自动关闭 eval/save 避免报错
    if max_steps <= 10:
        args_dict["eval_strategy"] = "no"
        args_dict["save_strategy"] = "no"

    # 确保有合理默认值
    args_dict.setdefault("dataloader_drop_last", True)
    args_dict.setdefault("remove_unused_columns", False)

    return TrainingArguments(
        output_dir=output_dir,
        **args_dict,
    )


def run_pretrain(
    config_path: str,
    data_path: str,
    tokenizer_path: str,
    output_dir: str,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> None:
    """执行预训练流程

    对应 from-scratch 版的 PreTrainer 完整训练流程：
    1. 加载配置、模型、tokenizer、数据
    2. 构建 Trainer 并训练
    3. 保存模型

    Args:
        config_path:  YAML 配置文件路径
        data_path:    预训练数据 JSONL 路径
        tokenizer_path: tokenizer 目录路径
        output_dir:   输出目录
        max_steps:    覆盖 config 中的 max_steps（可选）
        resume_from_checkpoint: 从 checkpoint 恢复训练（可选）
    """
    # 1. 加载 YAML 配置
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    pretrain_config = raw_config["pretrain"]

    # 2. 创建模型
    model_config = ClearMindConfig.from_yaml(config_path)
    model = ClearMindForCausalLM(model_config)

    # 3. 加载 tokenizer
    tokenizer = ClearMindTokenizer.load(tokenizer_path)

    # 4. 同步 vocab_size — tokenizer 实际大小可能与 config 不同
    if tokenizer.vocab_size != model_config.vocab_size:
        print(
            f"  vocab_size 不一致: config={model_config.vocab_size}, "
            f"tokenizer={tokenizer.vocab_size}"
        )
        print(f"  自动调整为 tokenizer 的 vocab_size={tokenizer.vocab_size}")
        model_config.vocab_size = tokenizer.vocab_size
        model = ClearMindForCausalLM(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 5. 加载数据
    max_length = raw_config["model"].get("max_position_embeddings", 512)
    datasets = load_pretrain_dataset(data_path, tokenizer, max_length=max_length)
    print(f"训练集: {len(datasets['train'])} 条, 验证集: {len(datasets['validation'])} 条")

    # 6. 创建 DataCollator — 自动生成 labels (input_ids 右移一位)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 7. 覆盖 max_steps（如果指定）
    if max_steps is not None:
        pretrain_config["max_steps"] = max_steps

    # 8. 构建 TrainingArguments 和 Trainer
    training_args = create_training_args(pretrain_config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        callbacks=[ClearMindLoggingCallback()],
    )

    # 9. 训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 10. 保存最终模型（HF 格式: config.json + model.safetensors）
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n模型已保存到: {output_dir}")

"""
dpo.py — DPO 对齐训练核心逻辑 (TRL DPOTrainer 版)
====================================================

使用 TRL DPOTrainer 执行 Direct Preference Optimization，
让模型学会偏好高质量回复、回避低质量回复。

from-scratch 对比:
  - DPOTrainer(BaseTrainer) 手写 DPO loss → TRL DPOTrainer 内置
  - copy.deepcopy(model) 创建 ref_model → DPOTrainer 自动管理
  - 手动计算 log_softmax + gather + mask → DPOTrainer 内置 log prob 计算
  - 手动 tokenize chosen/rejected → DPOTrainer 自动 tokenize

DPO 公式:
  L = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

这是大语言模型训练的第三阶段:
  Pre-training → SFT → DPO
"""

import copy

import yaml
from trl import DPOConfig, DPOTrainer

from model import ClearMindConfig, ClearMindForCausalLM
from data.tokenizer import ClearMindTokenizer
from data.data_utils import load_dpo_dataset
from training.callbacks import ClearMindLoggingCallback


def create_dpo_args(dpo_config: dict, output_dir: str) -> DPOConfig:
    """从 YAML dpo 节构建 DPOConfig

    DPOConfig 继承自 TrainingArguments，额外支持 beta 等 DPO 特有参数。

    tiny.yaml dpo 节字段: per_device_train_batch_size, gradient_accumulation_steps,
    num_train_epochs, learning_rate, beta, logging_steps, bf16, report_to

    Args:
        dpo_config: YAML 中 dpo 节的字典
        output_dir: 输出目录

    Returns:
        DPOConfig 实例
    """
    args_dict = dict(dpo_config)

    # DPO 合理默认值
    args_dict.setdefault("lr_scheduler_type", "cosine")
    args_dict.setdefault("warmup_ratio", 0.1)
    args_dict.setdefault("save_strategy", "epoch")
    args_dict.setdefault("eval_strategy", "epoch")
    args_dict.setdefault("gradient_checkpointing", False)
    args_dict.setdefault("remove_unused_columns", False)

    return DPOConfig(
        output_dir=output_dir,
        **args_dict,
    )


def run_dpo(
    config_path: str,
    data_path: str,
    tokenizer_path: str,
    output_dir: str,
    sft_model_path: str | None = None,
    num_train_epochs: int | None = None,
    max_steps: int | None = None,
) -> None:
    """执行 DPO 偏好对齐训练

    对应 from-scratch 版 DPOTrainer.train() 的完整流程。
    TRL DPOTrainer 自动处理:
      - ref_model 创建和冻结 (from-scratch: copy.deepcopy)
      - prompt/chosen/rejected tokenize (from-scratch: 手动 tokenize 4 组 tensor)
      - log prob 计算 (from-scratch: log_softmax + gather + mask)
      - DPO loss 计算 (from-scratch: -logsigmoid(beta * margin).mean())

    Args:
        config_path:       YAML 配置文件路径
        data_path:         DPO 数据 JSONL 路径
        tokenizer_path:    tokenizer 目录路径
        output_dir:        输出目录
        sft_model_path:    SFT 模型路径 (HF 格式目录, 可选)
        num_train_epochs:  覆盖 epoch 数 (可选)
        max_steps:         覆盖为 step-based 训练 (可选, 用于测试)
    """
    # 1. 加载 YAML 配置
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    dpo_config = raw_config.get("dpo", {})

    # 2. 加载 tokenizer
    tokenizer = ClearMindTokenizer.load(tokenizer_path)
    # DPOTrainer 需要 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 创建或加载模型
    if sft_model_path:
        print(f"从 SFT 模型加载: {sft_model_path}")
        model = ClearMindForCausalLM.from_pretrained(sft_model_path)
    else:
        print("未指定 SFT 模型, 从随机初始化开始 DPO")
        model_config = ClearMindConfig.from_yaml(config_path)
        if tokenizer.vocab_size != model_config.vocab_size:
            model_config.vocab_size = tokenizer.vocab_size
        model = ClearMindForCausalLM(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {num_params:,}")

    # 4. 加载 DPO 数据 — 字符串格式 {prompt, chosen, rejected}
    max_length = model.config.max_position_embeddings
    datasets = load_dpo_dataset(data_path, tokenizer, max_length=max_length)
    print(f"训练集: {len(datasets['train'])} 条, 验证集: {len(datasets['validation'])} 条")

    # 5. 覆盖参数
    if num_train_epochs is not None:
        dpo_config["num_train_epochs"] = num_train_epochs
    if max_steps is not None:
        dpo_config["max_steps"] = max_steps
        dpo_config["eval_strategy"] = "no"
        dpo_config["save_strategy"] = "no"

    # 6. 创建 ref_model — 冻结的 model 副本
    # from-scratch 对比: ref_model = copy.deepcopy(model)
    # DPOTrainer 可自动创建，但对无 name_or_path 的自定义模型需显式传入
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    # 7. 构建 DPOConfig 和 DPOTrainer
    # 设置 max_length 避免超过模型的 max_position_embeddings
    dpo_config.setdefault("max_length", max_length)
    training_args = create_dpo_args(dpo_config, output_dir)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        callbacks=[ClearMindLoggingCallback()],
    )

    # 8. 训练 — DPOTrainer 自动:
    #   - 创建 ref_model (冻结的 model 副本)
    #   - tokenize prompt + chosen/rejected
    #   - 计算 log probs 和 DPO loss
    trainer.train()

    # 9. 保存最终模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nDPO 模型已保存到: {output_dir}")

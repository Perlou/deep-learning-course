"""
pretrain.py — 预训练 Trainer
==============================

实现 GPT 模型的预训练流程 (Pre-training)。

预训练目标: Next-token Prediction
  给定前面的 token 序列, 预测下一个 token。
  Loss = CrossEntropy(model(x[:-1]), x[1:])

训练策略:
  - 优化器: AdamW (weight decay 解耦的 Adam)
  - 学习率: Cosine Annealing with Linear Warmup
  - 梯度累积: 模拟大 batch size
  - 梯度裁剪: 防止梯度爆炸
  - 混合精度: 可选 (MPS/CUDA)

这是大语言模型训练的第一阶段:
  Pre-training → SFT → DPO
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer_utils import (
    get_device,
    get_dtype,
    CosineWarmupScheduler,
    clip_grad_norm,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
)


class PreTrainer:
    """预训练 Trainer

    Args:
        model:       GPT 模型
        train_dataset: 预训练数据集
        config:      训练配置字典
        output_dir:  输出目录
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        output_dir: str = "outputs/pretrain",
    ):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设备和精度
        self.device = get_device()
        self.dtype = get_dtype(self.device, config.get("dtype", "float32"))
        print(f"🖥️  设备: {self.device}, 精度: {self.dtype}")

        # 模型
        self.model = model.to(self.device)

        # 数据加载器
        self.batch_size = config.get("batch_size", 16)
        self.gradient_accumulation = config.get("gradient_accumulation", 4)
        self.effective_batch_size = self.batch_size * self.gradient_accumulation

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # MacBook 上 num_workers>0 可能有问题
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
        )

        # 优化器: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

        # 学习率调度器
        self.max_steps = config.get("max_steps", 10000)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=config.get("lr", 3e-4),
            min_lr=config.get("min_lr", 3e-5),
            warmup_steps=config.get("warmup_steps", 500),
            max_steps=self.max_steps,
        )

        # Checkpoint
        self.save_every = config.get("save_every", 1000)
        self.eval_every = config.get("eval_every", 500)

        # 日志
        self.logger = TrainingLogger(
            log_dir=os.path.join(output_dir, "logs"),
            log_every=config.get("log_every", 10),
        )

        print(f"\n📋 预训练配置:")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Effective batch: {self.effective_batch_size}")
        print(f"  Max steps:       {self.max_steps}")
        print(
            f"  LR:              {config.get('lr', 3e-4)} → {config.get('min_lr', 3e-5)}"
        )
        print(f"  Warmup steps:    {config.get('warmup_steps', 500)}")
        print(f"  Save every:      {self.save_every} steps")

    def train(self, resume_from: str = None):
        """执行预训练

        Args:
            resume_from: Checkpoint 路径 (用于断点续训)
        """
        start_step = 0

        # 断点续训
        if resume_from and os.path.exists(resume_from):
            info = load_checkpoint(
                self.model,
                resume_from,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            start_step = info["step"]
            print(f"🔄 从 step {start_step} 恢复训练")

        print(f"\n{'=' * 60}")
        print(f"🚀 开始预训练 (step {start_step} → {self.max_steps})")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = start_step
        data_iter = iter(self.train_loader)
        accumulation_loss = 0.0

        while step < self.max_steps:
            # ========== 梯度累积循环 ==========
            self.optimizer.zero_grad()
            batch_loss = 0.0

            for micro_step in range(self.gradient_accumulation):
                # 获取数据 (循环 DataLoader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.dtype != torch.float32),
                ):
                    logits, loss = self.model(input_ids, labels)

                # 梯度累积: loss 除以累积步数
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()
                batch_loss += loss.item()

            # ========== 参数更新 ==========
            # 梯度裁剪
            grad_norm = clip_grad_norm(self.model, max_norm=1.0)

            # 优化器步进
            self.optimizer.step()

            # 学习率更新
            lr = self.scheduler.step()

            # ========== 日志记录 ==========
            avg_loss = batch_loss / self.gradient_accumulation
            tokens_per_step = self.effective_batch_size * input_ids.shape[1]

            self.logger.log(
                step=step,
                max_steps=self.max_steps,
                loss=avg_loss,
                lr=lr,
                tokens_per_step=tokens_per_step,
                grad_norm=grad_norm,
            )

            # ========== Checkpoint ==========
            if (step + 1) % self.save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step + 1,
                    loss=avg_loss,
                    save_path=os.path.join(
                        self.output_dir, f"checkpoint_step{step + 1}.pth"
                    ),
                )

            step += 1

        # ========== 训练结束 ==========
        # 保存最终模型
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            loss=avg_loss,
            save_path=os.path.join(self.output_dir, "final.pth"),
        )

        # 保存日志和摘要
        self.logger.save_log("pretrain_log.jsonl")
        self.logger.summary()

        print(f"\n{'=' * 60}")
        print(f"✅ 预训练完成! 共 {step} 步")
        print(f"{'=' * 60}")

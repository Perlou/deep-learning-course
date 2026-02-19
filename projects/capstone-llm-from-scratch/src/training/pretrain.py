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
  - 混合精度: 可选 (MPS/CUDA) + GradScaler
  - 验证集 + Early Stopping: 防止过拟合
  - TensorBoard: 可选训练可视化

这是大语言模型训练的第一阶段:
  Pre-training → SFT → DPO
"""

import os

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from .trainer_utils import (
    CosineWarmupScheduler,
    load_checkpoint,
)


class PreTrainer(BaseTrainer):
    """预训练 Trainer

    Args:
        model:         GPT 模型
        train_dataset: 预训练数据集
        val_dataset:   验证集 (可选, 用于 Early Stopping)
        config:        训练配置字典
        output_dir:    输出目录
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
        output_dir: str = "outputs/pretrain",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            output_dir=output_dir,
            stage_name="预训练",
            default_lr=3e-4,
            default_batch_size=16,
            default_grad_accum=4,
            default_patience=5,
            default_log_every=10,
        )

        # 预训练特有: step-based 训练
        self.max_steps = config.get("max_steps", 10000)
        self.eval_every = config.get("eval_every", 500)

        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=config.get("min_lr", self.lr * 0.1),
            warmup_steps=config.get("warmup_steps", 500),
            max_steps=self.max_steps,
        )

        print(f"\n📋 预训练配置:")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Effective batch: {self.effective_batch_size}")
        print(f"  Max steps:       {self.max_steps}")
        print(f"  LR:              {self.lr} → {config.get('min_lr', self.lr * 0.1)}")
        print(f"  Warmup steps:    {config.get('warmup_steps', 500)}")
        print(f"  Save every:      {self.save_every} steps")
        print(f"  Eval every:      {self.eval_every} steps")
        print(f"  Validation:      {'✅' if self.val_loader else '❌ (无验证集)'}")
        print(f"  GradScaler:      {'✅' if self.scaler else '❌'}")

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

                # 前向传播 (混合精度)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.dtype != torch.float32),
                ):
                    logits, loss, _ = self.model(input_ids, labels)

                # 梯度累积: loss 除以累积步数
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batch_loss += loss.item()

            # ========== 参数更新 ==========
            grad_norm, lr = self._optimizer_step()

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

            # ========== 验证 + Early Stopping ==========
            if self.val_loader and (step + 1) % self.eval_every == 0:
                val_loss = self._validate_loss(step)
                if self._check_early_stopping(val_loss, step + 1, "val_loss"):
                    break

            # ========== Checkpoint ==========
            if (step + 1) % self.save_every == 0:
                self._save(
                    step + 1,
                    avg_loss,
                    f"checkpoint_step{step + 1}.pth",
                )

            step += 1

        # ========== 训练结束 ==========
        self._finalize(step, avg_loss, "pretrain_log.jsonl")

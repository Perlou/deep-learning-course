"""
sft.py — SFT 指令微调 Trainer
==============================

在预训练模型上进行 Supervised Fine-Tuning (SFT)，
让模型从"续写文本"变成"按指令回答问题"。

SFT 与预训练的关键区别:
  1. 只在 Assistant 回复部分计算 Loss (由 labels mask 控制)
  2. 使用更小的学习率 (避免灾难性遗忘)
  3. 训练 epoch 较少 (通常 1-3 个 epoch)

这是大语言模型训练的第二阶段:
  Pre-training → SFT → DPO
"""

import math
import os

import torch
import torch.nn as nn

from .base_trainer import BaseTrainer
from .trainer_utils import (
    CosineWarmupScheduler,
    load_checkpoint,
    amp_autocast,
)


class SFTTrainer(BaseTrainer):
    """SFT 指令微调 Trainer

    Args:
        model:         GPT 模型
        train_dataset: SFT 数据集
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
        output_dir: str = "outputs/sft",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            output_dir=output_dir,
            stage_name="SFT 训练",
            default_lr=1e-5,
            default_batch_size=8,
            default_grad_accum=4,
            default_patience=3,
            default_log_every=10,
        )

        # SFT 特有: epoch-based 训练
        self.epochs = config.get("epochs", 3)
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation)
        )
        self.max_steps = self.steps_per_epoch * self.epochs

        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
            warmup_steps=min(100, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        print(f"\n📋 SFT 配置:")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Steps/epoch:     {self.steps_per_epoch}")
        print(f"  Total steps:     {self.max_steps}")
        print(f"  LR:              {self.lr}")
        print(f"  Validation:      {'✅' if self.val_loader else '❌ (无验证集)'}")
        print(f"  GradScaler:      {'✅' if self.scaler else '❌'}")

    def train(self, pretrained_path: str = None):
        """执行 SFT 训练

        Args:
            pretrained_path: 预训练 checkpoint 路径 (加载预训练权重)
        """
        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            load_checkpoint(self.model, pretrained_path, device=self.device)
            print(f"✅ 已加载预训练权重: {pretrained_path}")
        else:
            print("⚠️  未加载预训练权重, 从随机初始化开始 SFT")

        print(f"\n{'=' * 60}")
        print(f"🚀 开始 SFT 训练 ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = 0
        stopped_early = False
        avg_epoch_loss = 0.0

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_steps = 0

            self.optimizer.zero_grad()
            micro_count = 0
            micro_loss_sum = 0.0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播 (混合精度)
                with amp_autocast(self.device, self.dtype):
                    logits, loss, _ = self.model(input_ids, labels)

                # 梯度累积
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                micro_count += 1
                micro_loss_sum += loss.item()

                if micro_count >= self.gradient_accumulation:
                    # 参数更新
                    grad_norm, lr = self._optimizer_step()
                    avg_step_loss = micro_loss_sum / micro_count

                    # 日志
                    self.logger.log(
                        step=step,
                        max_steps=self.max_steps,
                        loss=avg_step_loss,
                        lr=lr,
                        grad_norm=grad_norm,
                    )

                    epoch_loss += avg_step_loss
                    epoch_steps += 1
                    step += 1
                    micro_count = 0
                    micro_loss_sum = 0.0

                    if step >= self.max_steps:
                        break

            # 处理 epoch 尾部不足梯度累积步数的剩余 micro-batch
            if 0 < micro_count and step < self.max_steps:
                self._rescale_grads_for_remainder(micro_count)
                grad_norm, lr = self._optimizer_step()
                avg_step_loss = micro_loss_sum / micro_count

                self.logger.log(
                    step=step,
                    max_steps=self.max_steps,
                    loss=avg_step_loss,
                    lr=lr,
                    grad_norm=grad_norm,
                )

                epoch_loss += avg_step_loss
                epoch_steps += 1
                step += 1

            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                print(f"  Epoch {epoch + 1} 平均 loss: {avg_epoch_loss:.4f}")

            # ========== 验证 + Early Stopping (每 epoch) ==========
            if self.val_loader:
                val_loss = self._validate_loss(step)
                if self._check_early_stopping(val_loss, step, "val_loss"):
                    stopped_early = True
                    break

            # 每个 epoch 结束保存 checkpoint
            self._save(
                step,
                avg_epoch_loss,
                f"epoch{epoch + 1}.pth",
            )

            if step >= self.max_steps:
                break

        # ========== 训练结束 ==========
        self._finalize(step, avg_epoch_loss, "sft_log.jsonl", stopped_early)

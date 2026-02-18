"""
sft.py — SFT 指令微调 Trainer
==============================

在预训练模型上进行 Supervised Fine-Tuning (SFT)，
让模型从"续写文本"变成"按指令回答问题"。

SFT 与预训练的关键区别:
  1. 数据格式: 对话数据 (Human + Assistant) 而非纯文本
  2. Loss Mask: 只对 Assistant 回复部分计算 loss
  3. 学习率: 更小 (1e-5 vs 3e-4), 避免灾难性遗忘
  4. 训练时长: 更短 (几个 epoch vs 几万 steps)

这是大语言模型训练的第二阶段:
  Pre-training → [SFT] → DPO
"""

import os
import sys

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


class SFTTrainer:
    """SFT 指令微调 Trainer

    Args:
        model:         GPT 模型
        train_dataset: SFT 数据集
        config:        训练配置字典
        output_dir:    输出目录
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        output_dir: str = "outputs/sft",
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
        self.batch_size = config.get("batch_size", 8)
        self.gradient_accumulation = config.get("gradient_accumulation", 4)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 优化器 (使用比预训练更小的学习率)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-5),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

        # 训练参数
        self.epochs = config.get("epochs", 3)
        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation
        self.max_steps = self.steps_per_epoch * self.epochs

        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=config.get("lr", 1e-5),
            min_lr=config.get("lr", 1e-5) * 0.1,
            warmup_steps=min(100, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        # 日志
        self.logger = TrainingLogger(
            log_dir=os.path.join(output_dir, "logs"),
            log_every=config.get("log_every", 10),
        )

        print(f"\n📋 SFT 配置:")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Steps/epoch:     {self.steps_per_epoch}")
        print(f"  Total steps:     {self.max_steps}")
        print(f"  LR:              {config.get('lr', 1e-5)}")

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
            print(f"⚠️  未加载预训练权重, 从随机初始化开始 SFT")

        print(f"\n{'=' * 60}")
        print(f"🚀 开始 SFT 训练 ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = 0

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_steps = 0

            data_iter = iter(self.train_loader)
            self.optimizer.zero_grad()
            micro_count = 0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.dtype != torch.float32),
                ):
                    logits, loss = self.model(input_ids, labels)

                # 梯度累积
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()
                micro_count += 1

                if micro_count >= self.gradient_accumulation:
                    # 参数更新
                    grad_norm = clip_grad_norm(self.model, max_norm=1.0)
                    self.optimizer.step()
                    lr = self.scheduler.step()
                    self.optimizer.zero_grad()
                    micro_count = 0

                    # 日志
                    self.logger.log(
                        step=step,
                        max_steps=self.max_steps,
                        loss=loss.item(),
                        lr=lr,
                        grad_norm=grad_norm,
                    )

                    epoch_loss += loss.item()
                    epoch_steps += 1
                    step += 1

                    if step >= self.max_steps:
                        break

            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                print(f"  Epoch {epoch + 1} 平均 loss: {avg_epoch_loss:.4f}")

            # 每个 epoch 结束保存 checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=step,
                loss=avg_epoch_loss if epoch_steps > 0 else 0.0,
                save_path=os.path.join(self.output_dir, f"epoch{epoch + 1}.pth"),
            )

        # 保存最终模型
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            loss=avg_epoch_loss if epoch_steps > 0 else 0.0,
            save_path=os.path.join(self.output_dir, "final.pth"),
        )

        self.logger.save_log("sft_log.jsonl")
        self.logger.summary()

        print(f"\n{'=' * 60}")
        print(f"✅ SFT 训练完成! 共 {step} 步")
        print(f"{'=' * 60}")

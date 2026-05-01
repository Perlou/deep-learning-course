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
from .lora import apply_lora, merge_lora, lora_state_dict


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

        # pad_token_id 用于构建 attention mask
        self.pad_token_id = config.get("pad_token_id", 0)

        # SFT 特有: epoch-based 训练
        self.epochs = config.get("epochs", 3)
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation)
        )
        # 默认按 epoch 算总步数；config.max_steps 可下调（用于本地快速冒烟，
        # 例如 tiny 想跑通 SFT 全流程但不想等 200k 个 batch）
        full_max = self.steps_per_epoch * self.epochs
        cfg_max = config.get("max_steps")
        self.max_steps = min(cfg_max, full_max) if cfg_max else full_max

        # NaN 守卫计数（A 方案 Layer 3）：被跳过的 micro-batch 数量
        self._nan_skipped = 0

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

    def _rebuild_optimizer(self):
        """LoRA 应用后重建 optimizer（只优化可训练参数）"""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.get("weight_decay", 0.01)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
        )
        # 重建 scheduler
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
            warmup_steps=min(100, self.max_steps // 10),
            max_steps=self.max_steps,
        )

    def train(self, pretrained_path: str = None):
        """执行 SFT 训练

        Args:
            pretrained_path: 预训练 checkpoint 路径 (加载预训练权重)

        优先级:
          1. ``<output_dir>/_resume.pth`` 存在 → 自动续训（显示 SFT 已经在跑）
          2. 否则加载 ``pretrained_path`` 作为初始权重（从预训练接续）
        """
        # ========== 0. 自动续训检查（优先级最高） ==========
        auto_info = self._try_auto_resume()
        start_epoch, start_step = 0, 0
        if auto_info is not None:
            start_epoch = auto_info.get("epoch", 0)
            start_step = auto_info.get("step", 0)
            print(f"🔄 自动从 epoch={start_epoch}, step={start_step} 续训 SFT")
        elif pretrained_path and os.path.exists(pretrained_path):
            # 1. 加载预训练权重（不带 optimizer/scheduler，仅模型权重）
            load_checkpoint(self.model, pretrained_path, device=self.device)
            print(f"✅ 已加载预训练权重: {pretrained_path}")
        else:
            print("⚠️  未加载预训练权重, 从随机初始化开始 SFT")

        # 应用 LoRA (在加载预训练权重之后)
        use_lora = self.config.get("lora", False)
        if use_lora:
            rank = self.config.get("lora_rank", 8)
            alpha = self.config.get("lora_alpha", 16.0)
            apply_lora(self.model, rank=rank, alpha=alpha)
            # LoRA 改变了可训练参数，需要重建 optimizer
            self._rebuild_optimizer()

        print(f"\n{'=' * 60}")
        print(f"🚀 开始 SFT 训练 ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = start_step
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
                attention_mask = (input_ids != self.pad_token_id).long()

                # 前向传播 (混合精度)
                with amp_autocast(self.device, self.dtype):
                    logits, loss, _ = self.model(input_ids, labels, attention_mask=attention_mask)

                loss_value = loss.item()

                # NaN/Inf 守卫（A 方案 Layer 3）：理论上 dataset+gpt 双层防御后不会触发，
                # 但保留兜底以应对 fp16 上溢、损坏样本等极端情况
                if not math.isfinite(loss_value):
                    self._nan_skipped += 1
                    if self._nan_skipped <= 5 or self._nan_skipped % 100 == 0:
                        print(
                            f"⚠️  跳过 NaN/Inf loss micro-batch "
                            f"(累计 {self._nan_skipped} 次)"
                        )
                    # 清掉这个 micro-batch 已经积累但还没 step 的梯度，避免污染下一组累积
                    self.optimizer.zero_grad(set_to_none=True)
                    micro_count = 0
                    micro_loss_sum = 0.0
                    continue

                # 梯度累积
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                micro_count += 1
                micro_loss_sum += loss_value

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
        use_lora = self.config.get("lora", False)
        if use_lora:
            # 保存 LoRA 权重
            lora_path = os.path.join(self.output_dir, "lora_weights.pth")
            torch.save(lora_state_dict(self.model), lora_path)
            print(f"💾 LoRA 权重已保存: {lora_path}")

            # 合并 LoRA 到原始权重后再保存 final.pth
            merge_lora(self.model)

        self._finalize(step, avg_epoch_loss, "sft_log.jsonl", stopped_early)

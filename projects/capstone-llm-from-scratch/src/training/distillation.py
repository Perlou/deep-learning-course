"""
distillation.py — 白盒蒸馏 Trainer
====================================

把大模型（teacher，如 ClearMind-Plus）的知识蒸馏到小模型（student，如 ClearMind-Base）。
做法是让 student 不仅学 ground-truth label，还要学 teacher 的整个 logit 分布。

Loss 公式（参考 minimind/trainer/train_distillation.py 与 Hinton 2015）：

    L = α · CE(student_logits, labels)
        + (1-α) · KL(softmax(student/T) || softmax(teacher/T)) · T²

  - **CE**：标准 cross-entropy，跟 SFT 一致，仅在 ``labels != -100`` 处计算
  - **KL**：student 学 teacher 的"软标签"分布，温度 T 控制分布平滑度
  - α=1 → 纯 SFT；α=0 → 纯蒸馏；典型用 α=0.1~0.3
  - T=1 → 普通 softmax；T=2~5 让 teacher 分布更平滑（暴露次优 token 信息）

典型用法：用 Plus（486M，已 SFT）作为 teacher，蒸馏出更强的 Base（68.8M）。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .trainer_utils import (
    CosineWarmupScheduler,
    load_checkpoint,
    amp_autocast,
)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(softmax(student/T) || softmax(teacher/T)) · T²

    实现细节：
      - ``F.kl_div`` 接受 log-probabilities 作 input、probabilities 作 target
      - 必须用 ``reduction="batchmean"``（不是默认的 mean，否则数值偏离 KL 标准定义）
      - 乘以 ``T²`` 是为了让 KL 项的梯度量级与 CE 一致（Hinton 论文）

    Args:
        student_logits: [N, V] 学生 logits（已展平 batch×seq → N）
        teacher_logits: [N, V] 教师 logits
        temperature:    温度 T

    Returns:
        scalar loss
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature**2)


class DistillationTrainer(BaseTrainer):
    """白盒蒸馏 Trainer

    Args:
        model:         student 模型（小，要训练的）
        train_dataset: SFT 风格数据集（与 SFTTrainer 一致）
        config:        训练配置字典
        val_dataset:   验证集
        output_dir:    输出目录

    Config 关键字段：
      - ``teacher_path``:    teacher checkpoint（必需）
      - ``alpha``:           CE 权重（默认 0.1）
      - ``temperature``:     蒸馏温度（默认 2.0）
      - ``epochs``、``batch_size``、``lr`` 等同 SFTTrainer
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
        output_dir: str = "outputs/distillation",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            output_dir=output_dir,
            stage_name="蒸馏训练",
            default_lr=1e-4,
            default_batch_size=8,
            default_grad_accum=4,
            default_patience=3,
            default_log_every=20,
        )

        self.alpha = float(config.get("alpha", 0.1))
        self.temperature = float(config.get("temperature", 2.0))
        self.teacher_path = config.get("teacher_path")
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.teacher_model: nn.Module | None = None  # 在 train() 中加载

        # SFT 风格 epoch-based 训练
        self.epochs = int(config.get("epochs", 3))
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation)
        )
        self.max_steps = self.steps_per_epoch * self.epochs

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
            warmup_steps=min(100, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        print("\n📋 蒸馏配置:")
        print(f"  α (CE 权重):     {self.alpha}")
        print(f"  T (温度):        {self.temperature}")
        print(f"  Teacher:         {self.teacher_path}")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Total steps:     {self.max_steps}")
        print(f"  LR:              {self.lr}")

    def _build_teacher(self, teacher_config_path: str | None = None) -> nn.Module:
        """加载 teacher 模型（与 student 共享架构或独立架构）"""
        from ..model.gpt import GPT
        from ..model.config import ModelConfig

        if teacher_config_path:
            # 用独立配置（典型场景：teacher=Plus 配置 + student=Base 配置）
            import yaml

            with open(teacher_config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            teacher_config = ModelConfig(**cfg_dict["model"])
        else:
            # 默认与 student 同架构（仅当不传 teacher_config 时；少见）
            teacher_config = self.model.config

        teacher = GPT(teacher_config).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def train(
        self,
        teacher_path: str | None = None,
        teacher_config: str | None = None,
        pretrained_path: str | None = None,
    ) -> None:
        """执行蒸馏训练

        Args:
            teacher_path:    teacher checkpoint（覆盖 config.teacher_path）
            teacher_config:  teacher 模型配置 yaml（teacher 与 student 架构不同时必传）
            pretrained_path: student 起点权重（默认从随机初始化或自动 resume）
        """
        # ---- 0. 自动续训 ----
        auto_info = self._try_auto_resume()
        start_step = 0
        if auto_info is not None:
            start_step = auto_info.get("step", 0)
            print(f"🔄 自动从 step={start_step} 续训蒸馏")
        elif pretrained_path and os.path.exists(pretrained_path):
            load_checkpoint(self.model, pretrained_path, device=self.device)
            print(f"✅ Student 已加载 pretrained: {pretrained_path}")

        # ---- 1. 加载 teacher ----
        teacher_path = teacher_path or self.teacher_path
        if not teacher_path or not os.path.exists(teacher_path):
            raise ValueError(
                f"蒸馏需要 teacher checkpoint，但 {teacher_path!r} 不存在。\n"
                "请用 --teacher 指定，或在 config 里设 teacher_path。"
            )

        print("\n📚 加载 teacher 模型 ...")
        self.teacher_model = self._build_teacher(teacher_config)
        load_checkpoint(self.teacher_model, teacher_path, device=self.device)
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"   Teacher params: {teacher_params / 1e6:.1f}M, "
            f"Student params: {student_params / 1e6:.1f}M, "
            f"压缩比: {teacher_params / student_params:.1f}x"
        )

        # 校验 vocab 一致（蒸馏要求两个模型用同一份 tokenizer）
        if self.teacher_model.config.vocab_size != self.model.config.vocab_size:
            raise ValueError(
                f"Teacher vocab_size ({self.teacher_model.config.vocab_size}) ≠ "
                f"Student vocab_size ({self.model.config.vocab_size})；"
                "蒸馏要求 teacher 和 student 共享 tokenizer/vocab"
            )

        print(f"\n{'=' * 60}")
        print(f"🚀 开始蒸馏 ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = start_step
        stopped_early = False
        avg_epoch_loss = 0.0

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_ce = 0.0
            epoch_kl = 0.0
            epoch_steps = 0

            self.optimizer.zero_grad(set_to_none=True)
            micro_count = 0
            micro_loss_sum = 0.0
            micro_ce_sum = 0.0
            micro_kl_sum = 0.0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = (input_ids != self.pad_token_id).long()

                with amp_autocast(self.device, self.dtype):
                    # Student 前向
                    s_logits, ce_loss, _ = self.model(
                        input_ids, labels, attention_mask=attention_mask
                    )
                    # Teacher 前向（no_grad）
                    with torch.no_grad():
                        t_logits, _, _ = self.teacher_model(
                            input_ids, attention_mask=attention_mask
                        )

                    # Shift（与训练 next-token-prediction 对齐）
                    s_shift = s_logits[..., :-1, :].contiguous()
                    t_shift = t_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # 只在 valid 位置（labels != -100）计算 KL
                    loss_mask = (shift_labels != -100).view(-1)
                    s_flat = s_shift.view(-1, s_shift.size(-1))[loss_mask]
                    t_flat = t_shift.view(-1, t_shift.size(-1))[loss_mask]

                    if s_flat.numel() == 0:
                        # 整个 batch 都是 padding，跳过
                        continue

                    kl_loss = distillation_loss(s_flat, t_flat, self.temperature)
                    loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

                scaled = loss / self.gradient_accumulation
                if self.scaler:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                micro_count += 1
                micro_loss_sum += loss.item()
                micro_ce_sum += ce_loss.item()
                micro_kl_sum += kl_loss.item()

                if micro_count >= self.gradient_accumulation:
                    grad_norm, lr = self._optimizer_step()
                    avg_loss = micro_loss_sum / micro_count
                    avg_ce = micro_ce_sum / micro_count
                    avg_kl = micro_kl_sum / micro_count

                    self.logger.log(
                        step=step,
                        max_steps=self.max_steps,
                        loss=avg_loss,
                        lr=lr,
                        grad_norm=grad_norm,
                    )
                    if self.logger.tb_writer:
                        self.logger.tb_writer.add_scalar("distill/ce", avg_ce, step)
                        self.logger.tb_writer.add_scalar("distill/kl", avg_kl, step)

                    if step % 5 == 0:
                        print(
                            f"  distill metrics: ce={avg_ce:.4f}, kl={avg_kl:.4f}, "
                            f"total={avg_loss:.4f}"
                        )

                    epoch_loss += avg_loss
                    epoch_ce += avg_ce
                    epoch_kl += avg_kl
                    epoch_steps += 1
                    step += 1
                    micro_count = 0
                    micro_loss_sum = micro_ce_sum = micro_kl_sum = 0.0

                    if step >= self.max_steps:
                        break

            # 处理 epoch 尾部不足 grad_accum 的剩余 micro-batch
            if 0 < micro_count and step < self.max_steps:
                self._rescale_grads_for_remainder(micro_count)
                grad_norm, lr = self._optimizer_step()
                epoch_loss += micro_loss_sum / micro_count
                epoch_ce += micro_ce_sum / micro_count
                epoch_kl += micro_kl_sum / micro_count
                epoch_steps += 1
                step += 1

            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                print(
                    f"  Epoch {epoch + 1} 平均: total={avg_epoch_loss:.4f}, "
                    f"ce={epoch_ce / epoch_steps:.4f}, "
                    f"kl={epoch_kl / epoch_steps:.4f}"
                )

            # 验证
            if self.val_loader:
                val_loss = self._validate_loss(step)
                if self._check_early_stopping(val_loss, step, "val_loss"):
                    stopped_early = True
                    break

            # 每 epoch 保存
            self._save(step, avg_epoch_loss, f"epoch{epoch + 1}.pth", epoch=epoch + 1)
            if step >= self.max_steps:
                break

        self._finalize(step, avg_epoch_loss, "distillation_log.jsonl", stopped_early)

        # 清理 teacher（释放显存）
        del self.teacher_model
        self.teacher_model = None

"""
dpo.py — DPO 对齐训练 Trainer
==============================

实现 Direct Preference Optimization (DPO)，
让模型学会偏好高质量回复、回避低质量回复。

DPO 核心思想:
  传统 RLHF 需要训练奖励模型 + PPO 强化学习，复杂且不稳定。
  DPO 直接用偏好数据优化策略模型, 等价于隐式奖励建模。

公式:
  L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中:
  - π_θ:   当前策略模型 (正在训练)
  - π_ref:  参考模型 (冻结的 SFT 模型)
  - y_w:    人类偏好的回复 (chosen)
  - y_l:    人类不偏好的回复 (rejected)
  - β:      温度参数 (控制偏离 SFT 的程度)

这是大语言模型训练的第三阶段:
  Pre-training → SFT → DPO
"""

import copy
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


class DPOTrainer(BaseTrainer):
    """DPO 对齐训练 Trainer

    Args:
        model:         GPT 模型 (将被训练)
        train_dataset: DPO 数据集
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
        output_dir: str = "outputs/dpo",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            output_dir=output_dir,
            stage_name="DPO 训练",
            default_lr=5e-6,
            default_batch_size=4,
            default_grad_accum=8,
            default_patience=3,
            default_log_every=5,
            early_stopping_mode="max",  # accuracy 越大越好
        )

        # DPO 特有参数
        self.beta = config.get("beta", 0.1)
        self.ref_model = None  # train() 中加载 SFT 权重后创建
        self.pad_token_id = config.get("pad_token_id", 0)

        # Epoch-based 训练
        self.epochs = config.get("epochs", 1)
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation)
        )
        self.max_steps = self.steps_per_epoch * self.epochs

        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
            warmup_steps=min(50, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        print(f"\n📋 DPO 配置:")
        print(f"  β (温度):       {self.beta}")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Total steps:     {self.max_steps}")
        print(f"  LR:              {self.lr}")
        print(f"  Validation:      {'✅' if self.val_loader else '❌ (无验证集)'}")
        print(f"  GradScaler:      {'✅' if self.scaler else '❌'}")

    # ----------------------------------------------------------
    # DPO 专有方法
    # ----------------------------------------------------------

    def _compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算模型在给定序列上的对数概率 (只计算 label 不为 -100 的位置)

        Args:
            model:     模型
            input_ids: 输入 token [batch, seq_len]
            labels:    标签 [batch, seq_len], -100 表示不计算

        Returns:
            对数概率之和 [batch]
        """
        attention_mask = (input_ids != self.pad_token_id).long()
        logits, _, _ = model(input_ids, attention_mask=attention_mask)  # [batch, seq, vocab]

        # Shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 计算每个 token 的对数概率
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # 只取目标 token 的概率
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0),
        ).squeeze(-1)

        # Mask: 只保留 label 不为 -100 的位置
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # 求和 (每个样本的总对数概率)
        return token_log_probs.sum(-1)

    def _dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """计算 DPO Loss

        L = -log σ(β · (log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)))

        Returns:
            (loss, metrics_dict)
        """
        chosen_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_ratio = policy_rejected_logps - ref_rejected_logps

        logits = self.beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            chosen_rewards = self.beta * chosen_ratio.detach()
            rejected_rewards = self.beta * rejected_ratio.detach()
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

        metrics = {
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def _evaluate_dpo(self) -> tuple[float, float]:
        """在验证集上评估 DPO 指标

        Returns:
            (val_loss, val_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        for batch in self.val_loader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)

            with amp_autocast(self.device, self.dtype):
                policy_chosen_logps = self._compute_log_probs(
                    self.model, chosen_ids, chosen_labels
                )
                policy_rejected_logps = self._compute_log_probs(
                    self.model, rejected_ids, rejected_labels
                )
                ref_chosen_logps = self._compute_log_probs(
                    self.ref_model, chosen_ids, chosen_labels
                )
                ref_rejected_logps = self._compute_log_probs(
                    self.ref_model, rejected_ids, rejected_labels
                )

            loss, metrics = self._dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            total_batches += 1

        self.model.train()
        n = max(total_batches, 1)
        return total_loss / n, total_accuracy / n

    # ----------------------------------------------------------
    # 训练循环
    # ----------------------------------------------------------

    def train(self, sft_path: str = None):
        """执行 DPO 训练

        Args:
            sft_path: SFT 模型 checkpoint 路径
        """
        # 加载 SFT 权重
        if sft_path and os.path.exists(sft_path):
            load_checkpoint(self.model, sft_path, device=self.device)
            print(f"✅ 已加载 SFT 权重: {sft_path}")
        else:
            print("⚠️  未加载 SFT 权重")

        # 创建参考模型 (冻结的 SFT 模型副本)
        print("📋 创建参考模型 (冻结)...")
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print(f"\n{'=' * 60}")
        print(f"🚀 开始 DPO 训练 ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = 0
        stopped_early = False
        epoch_loss = 0.0
        epoch_steps = 0

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_steps = 0

            self.optimizer.zero_grad()
            micro_count = 0
            micro_loss_sum = 0.0

            for batch in self.train_loader:
                # 取出 chosen 和 rejected 数据
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                with amp_autocast(self.device, self.dtype):
                    # 计算当前模型的 log probs
                    policy_chosen_logps = self._compute_log_probs(
                        self.model, chosen_ids, chosen_labels
                    )
                    policy_rejected_logps = self._compute_log_probs(
                        self.model, rejected_ids, rejected_labels
                    )

                    # 计算参考模型的 log probs (不计算梯度)
                    with torch.no_grad():
                        ref_chosen_logps = self._compute_log_probs(
                            self.ref_model, chosen_ids, chosen_labels
                        )
                        ref_rejected_logps = self._compute_log_probs(
                            self.ref_model, rejected_ids, rejected_labels
                        )

                    # DPO Loss
                    loss, metrics = self._dpo_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                    )

                # 梯度累积
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                micro_count += 1
                micro_loss_sum += loss.item()

                if micro_count >= self.gradient_accumulation:
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

                    # TensorBoard DPO 指标
                    if self.logger.tb_writer:
                        self.logger.tb_writer.add_scalar(
                            "dpo/accuracy", metrics["accuracy"], step
                        )
                        self.logger.tb_writer.add_scalar(
                            "dpo/reward_margin", metrics["reward_margin"], step
                        )

                    if step % 5 == 0:
                        print(
                            f"  DPO metrics: accuracy={metrics['accuracy']:.2%}, "
                            f"margin={metrics['reward_margin']:.4f}"
                        )

                    epoch_loss += avg_step_loss
                    epoch_accuracy += metrics["accuracy"]
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

                if self.logger.tb_writer:
                    self.logger.tb_writer.add_scalar("dpo/accuracy", metrics["accuracy"], step)
                    self.logger.tb_writer.add_scalar(
                        "dpo/reward_margin", metrics["reward_margin"], step
                    )

                epoch_loss += avg_step_loss
                epoch_accuracy += metrics["accuracy"]
                epoch_steps += 1
                step += 1

            if epoch_steps > 0:
                print(
                    f"  Epoch {epoch + 1}: loss={epoch_loss / epoch_steps:.4f}, "
                    f"accuracy={epoch_accuracy / epoch_steps:.2%}"
                )

            # ========== 验证 + Early Stopping (按 accuracy) ==========
            if self.val_loader:
                val_loss, val_accuracy = self._evaluate_dpo()
                print(
                    f"  📋 Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%} (step {step})"
                )

                if self.logger.tb_writer:
                    self.logger.tb_writer.add_scalar("val/loss", val_loss, step)
                    self.logger.tb_writer.add_scalar("val/accuracy", val_accuracy, step)

                if self._check_early_stopping(val_accuracy, step, "val_accuracy"):
                    stopped_early = True
                    break

            if step >= self.max_steps:
                break

        # ========== 训练结束 ==========
        final_loss = epoch_loss / max(epoch_steps, 1)
        self._finalize(step, final_loss, "dpo_log.jsonl", stopped_early)

        # 清理参考模型释放内存
        del self.ref_model
        self.ref_model = None

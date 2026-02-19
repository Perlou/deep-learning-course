"""
dpo.py — DPO 对齐训练 Trainer
==============================

实现 Direct Preference Optimization (DPO)，
让模型学会偏好高质量回复、回避低质量回复。

DPO 核心思想:
  给定 prompt, 让模型对 chosen (好) 回复的概率高于 rejected (差) 回复。
  不需要训练单独的 reward model (相比 RLHF 更简单)。

DPO Loss:
  L = -log σ(β · (log π_θ(chosen) - log π_ref(chosen)
                  - log π_θ(rejected) + log π_ref(rejected)))

其中:
  π_θ   = 当前训练中的模型
  π_ref = 参考模型 (冻结的 SFT 模型副本)
  β     = 温度参数, 控制偏离参考模型的程度
  σ     = Sigmoid 函数

训练改进:
  - 验证集 + Early Stopping (按 accuracy 监控)
  - 混合精度 GradScaler: CUDA FP16 防止梯度下溢
  - TensorBoard: 可选训练可视化

这是大语言模型训练的第三阶段:
  Pre-training → SFT → [DPO]
"""

import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .trainer_utils import (
    get_device,
    get_dtype,
    CosineWarmupScheduler,
    clip_grad_norm,
    save_checkpoint,
    load_checkpoint,
    TrainingLogger,
    EarlyStopping,
    create_grad_scaler,
)


class DPOTrainer:
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
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设备
        self.device = get_device()
        self.dtype = get_dtype(self.device, config.get("dtype", "float32"))
        print(f"🖥️  设备: {self.device}, 精度: {self.dtype}")

        # 训练模型 (π_θ)
        self.model = model.to(self.device)

        # 参考模型 (π_ref): SFT 模型的冻结副本
        # 在 train() 方法中加载 SFT 权重后创建
        self.ref_model = None

        # DPO 参数
        self.beta = config.get("beta", 0.1)

        # 数据
        self.batch_size = config.get("batch_size", 4)
        self.gradient_accumulation = config.get("gradient_accumulation", 8)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 验证集
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 5e-6),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

        # 训练参数
        self.epochs = config.get("epochs", 1)
        self.steps_per_epoch = len(self.train_loader) // self.gradient_accumulation
        self.max_steps = self.steps_per_epoch * self.epochs

        # 调度器
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=config.get("lr", 5e-6),
            min_lr=config.get("lr", 5e-6) * 0.1,
            warmup_steps=min(50, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        # GradScaler (CUDA FP16)
        self.scaler = create_grad_scaler(self.device, self.dtype)

        # Early Stopping (按 accuracy, 越大越好)
        self.early_stopping = None
        if self.val_loader is not None:
            self.early_stopping = EarlyStopping(
                patience=config.get("patience", 3),
                min_delta=config.get("min_delta", 0.0),
                mode="max",  # accuracy 越大越好
            )

        # 日志
        self.logger = TrainingLogger(
            log_dir=os.path.join(output_dir, "logs"),
            log_every=config.get("log_every", 5),
            use_tensorboard=config.get("use_tensorboard", False),
        )

        print(f"\n📋 DPO 配置:")
        print(f"  β (温度):       {self.beta}")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Grad accumulate: {self.gradient_accumulation}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Total steps:     {self.max_steps}")
        print(f"  LR:              {config.get('lr', 5e-6)}")
        print(f"  Validation:      {'✅' if self.val_loader else '❌ (无验证集)'}")
        print(f"  GradScaler:      {'✅' if self.scaler else '❌'}")

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
        logits, _, _ = model(input_ids)  # [batch, seq, vocab]

        # Shift: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 计算每个 token 的对数概率
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # 只取目标 token 的概率
        # gather: 按 labels 索引取值
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
        # 计算 log ratio
        chosen_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_ratio = policy_rejected_logps - ref_rejected_logps

        # DPO loss
        logits = self.beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()

        # 用于监控的指标
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

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=(self.dtype != torch.float32),
            ):
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

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_steps = 0

            self.optimizer.zero_grad()
            micro_count = 0

            for batch in self.train_loader:
                # 取出 chosen 和 rejected 数据
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.dtype != torch.float32),
                ):
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

                if micro_count >= self.gradient_accumulation:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_grad_norm(self.model, max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
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

                    epoch_loss += loss.item()
                    epoch_accuracy += metrics["accuracy"]
                    epoch_steps += 1
                    step += 1

                    if step >= self.max_steps:
                        break

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

                # 保存最优模型 (按 accuracy)
                if (
                    self.early_stopping.best_score is None
                    or val_accuracy > self.early_stopping.best_score
                ):
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step,
                        loss=val_loss,
                        save_path=os.path.join(self.output_dir, "best.pth"),
                    )
                    print(f"  💾 Best model saved (val_accuracy={val_accuracy:.2%})")

                # 检查是否应该停止
                if self.early_stopping(val_accuracy):
                    print(
                        f"\n⏹️  Early Stopping! patience={self.early_stopping.patience} 次未改善"
                    )
                    print(f"   最佳 val_accuracy: {self.early_stopping.best_score:.2%}")
                    stopped_early = True
                    break

        # 保存最终模型
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            loss=epoch_loss / max(epoch_steps, 1),
            save_path=os.path.join(self.output_dir, "final.pth"),
        )

        self.logger.save_log("dpo_log.jsonl")
        self.logger.summary()

        # 清理参考模型释放内存
        del self.ref_model
        self.ref_model = None

        print(f"\n{'=' * 60}")
        status = "Early Stopped" if stopped_early else "完成"
        print(f"✅ DPO 训练{status}! 共 {step} 步")
        print(f"{'=' * 60}")

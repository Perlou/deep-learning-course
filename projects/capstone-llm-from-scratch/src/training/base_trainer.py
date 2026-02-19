"""
base_trainer.py — Trainer 基类
================================

提取 PreTrainer / SFTTrainer / DPOTrainer 的公共逻辑:
  - 设备检测、精度设置
  - DataLoader 创建
  - 优化器、学习率调度器
  - GradScaler (CUDA FP16)
  - EarlyStopping
  - TrainingLogger + TensorBoard
  - 梯度累积 + 裁剪 + 参数更新
  - Checkpoint 保存

子类只需实现:
  - _setup_scheduler()  (可选覆盖)
  - _print_config()     配置打印
  - train()             训练循环
"""

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer_utils import (
    get_device,
    get_dtype,
    clip_grad_norm,
    save_checkpoint,
    TrainingLogger,
    EarlyStopping,
    evaluate_loss,
    create_grad_scaler,
)


class BaseTrainer(ABC):
    """Trainer 基类

    封装所有 Trainer 共享的初始化逻辑和工具方法。

    Args:
        model:         GPT 模型
        train_dataset: 训练数据集
        config:        训练配置字典
        val_dataset:   验证集 (可选)
        output_dir:    输出目录
        stage_name:    训练阶段名 (用于日志, 如 "预训练"/"SFT"/"DPO")
        default_lr:    默认学习率
        default_batch_size:     默认 batch size
        default_grad_accum:     默认梯度累积步数
        default_patience:       默认 Early Stopping patience
        default_log_every:      默认日志间隔
        early_stopping_mode:    "min" (loss 越小越好) 或 "max" (accuracy 越大越好)
        pin_memory:             是否使用 pin_memory (CUDA 加速)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
        output_dir: str = "outputs",
        stage_name: str = "训练",
        default_lr: float = 3e-4,
        default_batch_size: int = 16,
        default_grad_accum: int = 4,
        default_patience: int = 5,
        default_log_every: int = 10,
        early_stopping_mode: str = "min",
        pin_memory: bool = True,
    ):
        self.config = config
        self.output_dir = output_dir
        self.stage_name = stage_name
        os.makedirs(output_dir, exist_ok=True)

        # ========== 设备和精度 ==========
        self.device = get_device()
        self.dtype = get_dtype(self.device, config.get("dtype", "float32"))
        print(f"🖥️  设备: {self.device}, 精度: {self.dtype}")

        # ========== 模型 ==========
        self.model = model.to(self.device)

        # ========== 数据加载器 ==========
        self.batch_size = config.get("batch_size", default_batch_size)
        self.gradient_accumulation = config.get(
            "gradient_accumulation", default_grad_accum
        )
        self.effective_batch_size = self.batch_size * self.gradient_accumulation

        use_pin = pin_memory and self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=use_pin,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

        # ========== 优化器 ==========
        self.lr = config.get("lr", default_lr)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

        # ========== GradScaler ==========
        self.scaler = create_grad_scaler(self.device, self.dtype)

        # ========== Early Stopping ==========
        self.early_stopping = None
        if self.val_loader is not None:
            self.early_stopping = EarlyStopping(
                patience=config.get("patience", default_patience),
                min_delta=config.get("min_delta", 0.0),
                mode=early_stopping_mode,
            )

        # ========== Checkpoint ==========
        self.save_every = config.get("save_every", 1000)

        # ========== 日志 ==========
        self.logger = TrainingLogger(
            log_dir=os.path.join(output_dir, "logs"),
            log_every=config.get("log_every", default_log_every),
            use_tensorboard=config.get("use_tensorboard", False),
        )

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------

    def _backward_and_step(self, loss: torch.Tensor) -> tuple[float, float]:
        """执行反向传播 + 梯度裁剪 + 优化器更新 + 学习率更新

        Args:
            loss: 已经除以 gradient_accumulation 的 scaled loss

        Returns:
            (grad_norm, lr) 梯度范数和当前学习率
        """
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return 0.0, 0.0  # placeholder, real values returned by _optimizer_step

    def _optimizer_step(self) -> tuple[float, float]:
        """梯度裁剪 + 优化器更新 + 学习率更新

        Returns:
            (grad_norm, lr)
        """
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
        return grad_norm, lr

    def _save(self, step: int, loss: float, filename: str) -> None:
        """保存 checkpoint"""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            loss=loss,
            save_path=os.path.join(self.output_dir, filename),
        )

    def _validate_loss(self, step: int) -> float | None:
        """在验证集上评估 loss 并处理 early stopping

        Returns:
            val_loss 或 None (无验证集时)
        """
        if not self.val_loader:
            return None

        val_loss = evaluate_loss(self.model, self.val_loader, self.device, self.dtype)
        self.logger.log_val(step, val_loss)
        return val_loss

    def _check_early_stopping(
        self, metric: float, step: int, label: str = "val_loss"
    ) -> bool:
        """检查 early stopping 并保存最佳模型

        Args:
            metric: 验证指标
            step:   当前步数
            label:  指标名称 (用于打印)

        Returns:
            True 表示应该停止
        """
        if self.early_stopping is None:
            return False

        # 判断是否为新的最佳分数
        is_best = self.early_stopping.best_score is None
        if not is_best:
            if self.early_stopping.mode == "min":
                is_best = metric < self.early_stopping.best_score
            else:
                is_best = metric > self.early_stopping.best_score

        if is_best:
            self._save(step, metric, "best.pth")
            if self.early_stopping.mode == "max":
                print(f"  💾 Best model saved ({label}={metric:.2%})")
            else:
                print(f"  💾 Best model saved ({label}={metric:.4f})")

        if self.early_stopping(metric):
            print(
                f"\n⏹️  Early Stopping! patience={self.early_stopping.patience} 次未改善"
            )
            if self.early_stopping.mode == "max":
                print(f"   最佳 {label}: {self.early_stopping.best_score:.2%}")
            else:
                print(f"   最佳 {label}: {self.early_stopping.best_score:.4f}")
            return True

        return False

    def _finalize(
        self, step: int, loss: float, log_name: str, stopped_early: bool = False
    ) -> None:
        """训练结束的通用收尾逻辑"""
        self._save(step, loss, "final.pth")
        self.logger.save_log(log_name)
        self.logger.summary()

        status = "Early Stopped" if stopped_early else "完成"
        print(f"\n{'=' * 60}")
        print(f"✅ {self.stage_name}{status}! 共 {step} 步")
        print(f"{'=' * 60}")

    @abstractmethod
    def train(self, **kwargs) -> None:
        """执行训练循环 (子类必须实现)"""
        ...

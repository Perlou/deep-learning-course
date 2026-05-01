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
import random
from abc import ABC, abstractmethod

import numpy as np
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

        # ========== 可复现性种子 ==========
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        # ========== 设备和精度 ==========
        self.device = get_device()
        self.dtype = get_dtype(self.device, config.get("dtype", "float32"))
        print(f"🖥️  设备: {self.device}, 精度: {self.dtype}")

        # ========== 模型 ==========
        self.model = model.to(self.device)

        # ========== Activation Checkpointing（Phase 4） ==========
        # yaml.use_gradient_checkpointing=true 时启用：显存降 30-50%、吞吐 -25%
        # 对 plus（24 层 / 1280 hidden）训练时上下文比较紧的场景特别有用
        if config.get("use_gradient_checkpointing", False):
            inner = self.model.module if hasattr(self.model, "module") else self.model
            inner = getattr(inner, "_orig_mod", inner)
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable()
                print("💾 Activation checkpointing 已启用（显存↓ 吞吐↓）")

        # ========== torch.compile（CUDA bf16/fp16 + Python 3.11 才推荐） ==========
        # 对于 Plus 这类 ~500M 模型，单卡 +20-40% 训练速度
        # 风险：首次编译慢（30-90s）、少数算子（如自定义 RMSNorm）可能 fallback
        # 通过 yaml.use_compile=true 或 env CLEARMIND_COMPILE=1 启用
        use_compile = config.get("use_compile", False) or (
            os.environ.get("CLEARMIND_COMPILE", "").lower() in ("1", "true", "yes")
        )
        if use_compile and self.device.type == "cuda":
            try:
                compile_mode = config.get("compile_mode", "default")  # default / reduce-overhead / max-autotune
                self.model = torch.compile(self.model, mode=compile_mode)
                print(f"⚡ torch.compile 已启用（mode={compile_mode}）")
            except Exception as e:
                print(f"⚠️  torch.compile 失败，回退到 eager: {e}")
        elif use_compile:
            print(f"⚠️  torch.compile 已请求但 device={self.device.type}，仅 CUDA 支持，已跳过")

        # ========== 数据加载器 ==========
        self.batch_size = config.get("batch_size", default_batch_size)
        self.gradient_accumulation = config.get(
            "gradient_accumulation", default_grad_accum
        )
        self.effective_batch_size = self.batch_size * self.gradient_accumulation

        # DataLoader 默认参数（参考 minimind + 通用 best practice）：
        #   - num_workers: CUDA 默认 4，CPU/MPS 默认 0（避免多进程拷贝）
        #   - pin_memory: 仅 CUDA 启用（加速 H2D 拷贝）
        #   - persistent_workers: num_workers > 0 时启用（worker 不在每 epoch 后销毁）
        #   - prefetch_factor: 默认 2（每 worker 预取 2 个 batch）
        is_cuda = self.device.type == "cuda"
        use_pin = pin_memory and is_cuda
        num_workers = config.get("num_workers", 4 if is_cuda else 0)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": use_pin,
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)

        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

        self.val_loader = None
        if val_dataset is not None:
            val_kwargs = dict(loader_kwargs)
            val_kwargs["drop_last"] = False
            self.val_loader = DataLoader(val_dataset, shuffle=False, **val_kwargs)

        # ========== 优化器 (参数分组) ==========
        self.lr = config.get("lr", default_lr)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # 一维参数 (bias, RMSNorm gamma) 和 embedding 不做 weight decay
            if param.ndim == 1 or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # AdamW with fused kernel（CUDA + 支持的 dtype 自动启用，~10-15% 端到端加速）
        # PyTorch 2.4+ 的 AdamW 接受 fused=True；CPU/MPS 不支持，传 False
        # 如果 yaml 显式设了 use_fused_adamw=false 也尊重之
        use_fused = (
            config.get("use_fused_adamw", True)
            and self.device.type == "cuda"
        )
        adamw_kwargs = {
            "lr": self.lr,
            "betas": (config.get("beta1", 0.9), config.get("beta2", 0.95)),
        }
        if use_fused:
            try:
                # 试探性传 fused=True；旧版 torch 会 TypeError，自动回退
                adamw_kwargs["fused"] = True
            except Exception:
                pass

        try:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": config.get("weight_decay", 0.01)},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                **adamw_kwargs,
            )
            if use_fused:
                print("⚡ AdamW fused kernel 已启用")
        except (TypeError, RuntimeError) as e:
            # fused 不支持时回退
            adamw_kwargs.pop("fused", None)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": config.get("weight_decay", 0.01)},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                **adamw_kwargs,
            )
            if use_fused:
                print(f"⚠️  AdamW fused 不可用，回退普通版本: {e}")

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
        # wandb / swanlab 配置（可选）：
        #   yaml 中可设：
        #     pretrain:
        #       use_wandb: true
        #       wandb_backend: swanlab   # 或 wandb
        #       wandb_project: "ClearMind-Plus"
        #       wandb_run_name: null
        wandb_config = None
        if config.get("use_wandb", False):
            wandb_config = {
                "backend": config.get("wandb_backend", "swanlab"),
                "project": config.get("wandb_project", f"ClearMind-{stage_name}"),
                "run_name": config.get("wandb_run_name"),
                "tags": config.get("wandb_tags"),
                "config": {
                    k: v for k, v in config.items()
                    if isinstance(v, (int, float, str, bool, list))
                },
            }

        self.logger = TrainingLogger(
            log_dir=os.path.join(output_dir, "logs"),
            log_every=config.get("log_every", default_log_every),
            use_tensorboard=config.get("use_tensorboard", False),
            wandb_config=wandb_config,
        )

    # ----------------------------------------------------------
    # 工具方法
    # ----------------------------------------------------------

    def _optimizer_step(self) -> tuple[float, float]:
        """梯度裁剪 + 优化器更新 + 学习率更新

        关键修复：当 ``GradScaler`` 检测到梯度 inf/nan 跳过 ``optimizer.step()`` 时，
        **不应该** 调用 ``scheduler.step()``。否则 LR 进度被错误推进，
        实际 optimizer 没走但 LR 已经衰减，会导致后续训练曲线整体偏移。

        检测方式：``scaler.update()`` 后比较 scale 值是否变小。
          - 变小 → 当前 batch 触发了 inf/nan，optimizer 被跳过
          - 不变 / 变大 → optimizer 正常更新

        Returns:
            (grad_norm, lr)
        """
        optimizer_stepped = True

        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm(self.model, max_norm=1.0)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # scale 变小 = 触发了 inf/nan，optimizer 被跳过
            if self.scaler.get_scale() < scale_before:
                optimizer_stepped = False
        else:
            grad_norm = clip_grad_norm(self.model, max_norm=1.0)
            self.optimizer.step()

        # 仅当 optimizer 真正更新时才推进 scheduler
        if optimizer_stepped:
            lr = self.scheduler.step()
        else:
            # 跳过本次更新，LR 保持不变
            lr = self.scheduler.get_lr()

        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm, lr

    def _rescale_grads_for_remainder(self, micro_count: int) -> None:
        """修正不足梯度累积步数时的梯度缩放。

        常规路径中 loss 会除以 gradient_accumulation。
        当一个 epoch 末尾只剩余 micro_count(<gradient_accumulation) 个 micro-batch 时，
        这里把梯度按比例放大，避免最后一次更新被系统性缩小。
        """
        if micro_count <= 0 or micro_count >= self.gradient_accumulation:
            return

        scale = self.gradient_accumulation / micro_count
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    def _save(
        self,
        step: int,
        loss: float,
        filename: str,
        epoch: int = 0,
        save_resume: bool = True,
        half_weights: bool = True,
    ) -> None:
        """保存 checkpoint（双文件分离 + 原子写 + .half() 落盘）

        Args:
            step:         当前 optimizer 步数
            loss:         当前 loss
            filename:     纯权重文件名（如 ``"final.pth"`` / ``"checkpoint_step1000.pth"``）
            epoch:        当前 epoch（pretrain 是 step-based，可忽略）
            save_resume:  ``True`` 时同时写 ``_resume.pth``（中断恢复用）
            half_weights: ``True`` 时纯权重文件落盘前转 ``half().cpu()``
        """
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            loss=loss,
            save_path=os.path.join(self.output_dir, filename),
            epoch=epoch,
            scaler=self.scaler,
            save_resume=save_resume,
            half_weights=half_weights,
        )

    def _try_auto_resume(self) -> dict | None:
        """自动检测并加载 ``<output_dir>/_resume.pth``（如存在）

        在 train() 入口调用：
          - 找到 _resume.pth → 加载并返回 ``{"step": int, "epoch": int, "loss": float}``
          - 没找到 → 返回 None

        与显式 ``--resume <path>`` 的优先级关系由子类决定（一般是显式优先）。
        """
        from .trainer_utils import find_resume_checkpoint, load_checkpoint

        resume_path = find_resume_checkpoint(self.output_dir)
        if resume_path is None:
            return None
        print(f"\n🔄 自动续训：检测到 {resume_path}")
        info = load_checkpoint(
            self.model,
            resume_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )
        return info

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

"""
trainer_utils — 训练工具函数
============================

提供训练过程中通用的工具函数:
  - 学习率调度器 (Cosine with Warmup)
  - 梯度裁剪
  - Checkpoint 保存/加载
  - 训练日志记录
  - 设备检测
  - 验证 & Early Stopping
  - 混合精度 GradScaler
  - 多卡并行 (DDP)
  - TensorBoard 集成
"""

import os
import math
import time
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# 设备检测
# ============================================================


def get_device() -> torch.device:
    """自动检测最优计算设备

    优先级: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device, config_dtype: str = "float32") -> torch.dtype:
    """根据设备和配置获取计算精度

    Args:
        device:      计算设备
        config_dtype: 配置中指定的精度 ("float32", "float16", "bfloat16")
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config_dtype, torch.float32)

    # MPS 目前对 bfloat16 支持有限, 回退到 float32
    if device.type == "mps" and dtype == torch.bfloat16:
        print("⚠️  MPS 不支持 bfloat16, 回退到 float32")
        return torch.float32

    return dtype


# ============================================================
# 学习率调度器
# ============================================================


class CosineWarmupScheduler:
    """Cosine Annealing with Linear Warmup 学习率调度器

    学习率变化曲线:
      1. Warmup 阶段: 线性从 0 → max_lr
      2. Cosine 阶段: 余弦衰减从 max_lr → min_lr

    这是 GPT / Llama 等模型训练中标准的 LR 调度策略。

    Args:
        optimizer:    优化器
        max_lr:       最大学习率
        min_lr:       最小学习率
        warmup_steps: warmup 步数
        max_steps:    总训练步数
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def get_lr(self, step: Optional[int] = None) -> float:
        """计算当前步数的学习率"""
        if step is None:
            step = self.current_step

        if step < self.warmup_steps:
            # 线性 warmup: 0 → max_lr
            return self.max_lr * step / max(1, self.warmup_steps)
        elif step >= self.max_steps:
            return self.min_lr
        else:
            # Cosine annealing: max_lr → min_lr
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

    def step(self) -> float:
        """更新学习率"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr


# ============================================================
# Early Stopping
# ============================================================


class EarlyStopping:
    """Early Stopping 监控器

    跟踪验证指标, patience 次未改善则建议停止训练。

    Args:
        patience:  容忍多少次不改善
        min_delta: 改善幅度阈值 (loss 需下降 min_delta 才算改善)
        mode:      'min' 表示越小越好 (loss), 'max' 表示越大越好 (accuracy)
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """检查是否应该停止训练

        Args:
            metric: 当前验证指标

        Returns:
            True 表示应该停止
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


# ============================================================
# 验证评估
# ============================================================


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    val_loader,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_batches: int = None,
) -> float:
    """在验证集上计算平均 Loss

    Args:
        model:       GPT 模型
        val_loader:  验证集 DataLoader
        device:      计算设备
        dtype:       计算精度
        max_batches: 最多评估多少个 batch (None=全部)

    Returns:
        平均 loss
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            _, loss, _ = model(input_ids, labels)

        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


# ============================================================
# 混合精度 GradScaler
# ============================================================


def create_grad_scaler(
    device: torch.device, dtype: torch.dtype
) -> torch.amp.GradScaler | None:
    """为 CUDA FP16 创建 GradScaler

    只在 CUDA + float16 时启用 (bfloat16 不需要)。
    MPS / CPU 返回 None。

    Returns:
        GradScaler 或 None
    """
    if device.type == "cuda" and dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


# ============================================================
# 多卡数据并行 (DDP)
# ============================================================


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """初始化 DDP 进程组

    Args:
        rank:       当前进程的 rank
        world_size: 总进程数
        backend:    通信后端 ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """清理 DDP 进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """判断当前是否为主进程 (rank 0)

    非 DDP 环境下返回 True。
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def wrap_model_ddp(model: nn.Module, device_id: int = None) -> nn.Module:
    """可选地将模型包装为 DDP

    只在 DDP 已初始化时包装。

    Returns:
        原始模型或 DDP 包装的模型
    """
    if not dist.is_initialized():
        return model
    return DDP(model, device_ids=[device_id] if device_id is not None else None)


# ============================================================
# 梯度裁剪
# ============================================================


def clip_grad_norm(model: nn.Module, max_norm: float = 1.0) -> float:
    """梯度裁剪 (防止梯度爆炸)

    Args:
        model:    模型
        max_norm: 最大梯度范数

    Returns:
        裁剪前的梯度范数
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


# ============================================================
# Checkpoint 管理
# ============================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    save_path: str,
) -> None:
    """保存训练 checkpoint

    保存内容包括:
      - 模型权重
      - 优化器状态 (momentum, adaptive lr 等)
      - 调度器状态
      - 当前训练步数和 loss
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step": scheduler.current_step if scheduler else 0,
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint 保存: {save_path} (step={step}, loss={loss:.4f})")


def load_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None,
) -> dict:
    """加载训练 checkpoint

    Args:
        model:     模型
        save_path: checkpoint 文件路径
        optimizer: 优化器 (可选, 用于恢复训练)
        scheduler: 调度器 (可选)
        device:    加载到的设备

    Returns:
        checkpoint 信息字典
    """
    map_location = device if device else "cpu"
    checkpoint = torch.load(save_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"📦 模型权重加载: {save_path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"   优化器状态恢复")

    if scheduler and "scheduler_step" in checkpoint:
        scheduler.current_step = checkpoint["scheduler_step"]
        print(f"   调度器状态恢复 (step={scheduler.current_step})")

    info = {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
    }
    print(f"   恢复到 step={info['step']}, loss={info['loss']:.4f}")

    return info


# ============================================================
# 训练日志
# ============================================================


class TrainingLogger:
    """训练日志记录器

    记录并打印训练过程中的关键指标:
      - Loss, Learning Rate
      - 训练速度 (tokens/sec, steps/sec)
      - 预估剩余时间 (ETA)
      - TensorBoard 可视化 (可选)

    Args:
        log_dir:          日志保存目录
        log_every:        每多少步打印一次日志
        use_tensorboard:  是否启用 TensorBoard
    """

    def __init__(
        self, log_dir: str = None, log_every: int = 10, use_tensorboard: bool = False
    ):
        self.log_every = log_every
        self.log_dir = log_dir
        self.history = []
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.tb_writer = None

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(log_dir, "tensorboard") if log_dir else "runs"
                self.tb_writer = SummaryWriter(tb_dir)
                print(f"📊 TensorBoard 已启用: {tb_dir}")
            except ImportError:
                print("⚠️  tensorboard 未安装, 跳过 TensorBoard 集成")
                print("   安装: pip install tensorboard")

    def log(
        self,
        step: int,
        max_steps: int,
        loss: float,
        lr: float,
        tokens_per_step: int = 0,
        grad_norm: float = 0.0,
    ) -> None:
        """记录一步的训练指标"""
        now = time.time()
        elapsed = now - self.start_time
        step_time = now - self.step_start_time
        self.step_start_time = now

        entry = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_time": step_time,
            "elapsed": elapsed,
        }

        if tokens_per_step > 0:
            entry["tokens_per_sec"] = tokens_per_step / max(step_time, 1e-6)

        self.history.append(entry)

        # TensorBoard 写入
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, step)
            self.tb_writer.add_scalar("train/lr", lr, step)
            if grad_norm > 0:
                self.tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            if "tokens_per_sec" in entry:
                self.tb_writer.add_scalar(
                    "train/tokens_per_sec", entry["tokens_per_sec"], step
                )

        # 打印日志
        if step % self.log_every == 0 or step == max_steps - 1:
            eta = self._estimate_eta(step, max_steps, elapsed)

            msg = (
                f"Step {step:>6d}/{max_steps} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Grad: {grad_norm:.2f} | "
            )

            if "tokens_per_sec" in entry:
                msg += f"Speed: {entry['tokens_per_sec']:.0f} tok/s | "

            msg += f"ETA: {eta}"
            print(msg)

    def _estimate_eta(self, step: int, max_steps: int, elapsed: float) -> str:
        """估算剩余训练时间"""
        if step == 0:
            return "计算中..."

        avg_step_time = elapsed / step
        remaining_steps = max_steps - step
        remaining_seconds = avg_step_time * remaining_steps

        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds / 60:.1f}min"
        else:
            return f"{remaining_seconds / 3600:.1f}h"

    def save_log(self, filename: str = "training_log.jsonl") -> None:
        """保存训练日志到文件"""
        if not self.log_dir:
            return

        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            for entry in self.history:
                f.write(json.dumps(entry) + "\n")
        print(f"📝 训练日志保存: {filepath}")

    def log_val(self, step: int, val_loss: float) -> None:
        """记录验证指标"""
        print(f"  📋 Val Loss: {val_loss:.4f} (step {step})")
        if self.tb_writer:
            self.tb_writer.add_scalar("val/loss", val_loss, step)

    def summary(self) -> None:
        """打印训练摘要"""
        if not self.history:
            return

        total_time = time.time() - self.start_time
        total_steps = len(self.history)
        final_loss = self.history[-1]["loss"]
        min_loss = min(e["loss"] for e in self.history)

        print(f"\n📊 训练摘要:")
        print(f"  总步数:   {total_steps}")
        print(f"  总耗时:   {total_time / 60:.1f} 分钟")
        print(f"  最终 loss: {final_loss:.4f}")
        print(f"  最低 loss: {min_loss:.4f}")

        if any("tokens_per_sec" in e for e in self.history):
            avg_speed = (
                sum(e.get("tokens_per_sec", 0) for e in self.history) / total_steps
            )
            print(f"  平均速度: {avg_speed:.0f} tokens/sec")

        # 关闭 TensorBoard
        if self.tb_writer:
            self.tb_writer.close()

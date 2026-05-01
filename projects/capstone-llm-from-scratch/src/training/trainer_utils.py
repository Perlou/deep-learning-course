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

# PyTorch 版本兼容: torch.amp (2.1+) vs torch.cuda.amp (旧版)
try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler

    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore

    _AMP_NEW_API = False


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
    skipped = 0

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with amp_autocast(device, dtype):
            _, loss, _ = model(input_ids, labels)

        loss_value = loss.item()
        # NaN/Inf 守卫（A 方案 Layer 3）：避免单个坏 batch 污染验证均值
        if not math.isfinite(loss_value):
            skipped += 1
            continue
        total_loss += loss_value
        total_batches += 1

    if skipped > 0:
        print(f"⚠️  evaluate_loss 跳过 {skipped} 个 NaN/Inf batch")

    model.train()
    return total_loss / max(total_batches, 1)


def amp_autocast(device: torch.device, dtype: torch.dtype):
    """创建兼容所有 PyTorch 版本的 autocast 上下文管理器"""
    enabled = dtype != torch.float32
    if _AMP_NEW_API:
        return _autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    return _autocast(enabled=enabled)


# ============================================================
# 混合精度 GradScaler
# ============================================================


def create_grad_scaler(
    device: torch.device, dtype: torch.dtype
) -> Optional["_GradScaler"]:
    """为 CUDA FP16 创建 GradScaler

    只在 CUDA + float16 时启用 (bfloat16 不需要)。
    MPS / CPU 返回 None。

    Returns:
        GradScaler 或 None
    """
    if device.type == "cuda" and dtype == torch.float16:
        return _GradScaler("cuda") if _AMP_NEW_API else _GradScaler()
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
#
# 设计原则（参考 minimind/trainer/trainer_utils.py）：
#
# 1. **双文件分离**：每次保存写两份
#    - ``<save_path>``：纯模型权重（`.half().cpu()`），用于推理 / 发布
#    - ``<save_path 同目录>/_resume.pth``：完整训练状态（model + optimizer +
#      scheduler + scaler + step + epoch），用于断点续训
#
# 2. **原子写**：先写 ``<path>.tmp`` 再 ``os.replace``，保证训练中断不会留下损坏 ckpt
#
# 3. **`.half().cpu()` 落盘**：纯权重文件压缩到 fp16，磁盘占用减半，加载更快
#    （resume 文件保留 fp32 优化器状态，因为 Adam moments 必须高精度）
#
# 4. **GradScaler state 一并保存**：fp16 训练中断后 scaler.scale 必须恢复，
#    否则梯度溢出阈值会退回初始值，前几步又会触发 NaN 跳过更新
#
# 5. **加载兼容旧 ckpt**：旧 ckpt 缺少 scaler/epoch 字段时给合理默认（不报错）


def _atomic_torch_save(obj, path: str) -> None:
    """原子保存：写到 .tmp 再 os.replace，防止保存到一半挂掉"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _resume_path_for(save_path: str) -> str:
    """根据纯权重文件路径推导 _resume.pth 路径

    ``outputs/pretrain/final.pth`` → ``outputs/pretrain/_resume.pth``
    ``outputs/pretrain/checkpoint_step1000.pth`` → ``outputs/pretrain/_resume.pth``
    （所有 step checkpoint 共享同一个 _resume.pth，永远只保留最新一份）
    """
    return os.path.join(os.path.dirname(save_path), "_resume.pth")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    save_path: str,
    epoch: int = 0,
    scaler=None,
    save_resume: bool = True,
    half_weights: bool = True,
) -> None:
    """保存训练 checkpoint（原子写 + 双文件分离）

    Args:
        model:        模型
        optimizer:    优化器
        scheduler:    LR 调度器（如有 ``current_step`` 字段则保存）
        step:         当前 optimizer 更新步数
        loss:         当前 loss（仅记录用）
        save_path:    纯模型权重文件路径（如 ``outputs/sft/final.pth``）
        epoch:        当前 epoch（默认 0；pretrain 是 step-based，可忽略）
        scaler:       :class:`torch.amp.GradScaler` 实例（fp16 训练时必传）
        save_resume:  ``True`` 时同时写 ``_resume.pth``（中断恢复用）
        half_weights: ``True`` 时纯权重文件落盘前转 ``half().cpu()``（推理用）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 取出真实 module（兼容 DDP / torch.compile 包装）
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)

    # ---- 文件 1: 纯模型权重（.half().cpu()，用于推理/发布）----
    state_dict = raw_model.state_dict()
    if half_weights:
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    _atomic_torch_save(state_dict, save_path)
    size_mb = os.path.getsize(save_path) / 1024**2
    print(f"💾 模型权重保存: {save_path} (step={step}, loss={loss:.4f}, {size_mb:.1f}MB)")

    # ---- 文件 2: 完整训练状态（用于续训）----
    if save_resume:
        resume_path = _resume_path_for(save_path)
        # 注意：_resume 文件保留 fp32 优化器/scaler state，不做 half
        resume_state: dict = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": getattr(scheduler, "current_step", 0) if scheduler else 0,
            "step": step,
            "epoch": epoch,
            "loss": loss,
        }
        if scaler is not None and hasattr(scaler, "state_dict"):
            resume_state["scaler_state_dict"] = scaler.state_dict()
        _atomic_torch_save(resume_state, resume_path)
        rsize_mb = os.path.getsize(resume_path) / 1024**2
        print(f"💾 续训状态保存: {resume_path} ({rsize_mb:.1f}MB)")


def load_checkpoint(
    model: nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler=None,
    device: torch.device = None,
    strict: bool = False,
) -> dict:
    """加载训练 checkpoint

    自动判断文件类型：
      - 含 ``model_state_dict`` 字段 → 完整 resume 文件
      - 否则视为纯权重文件（``.half()`` 推理 ckpt）

    Args:
        model:     模型（接收 state_dict 加载）
        save_path: ckpt 文件路径（纯权重 *.pth 或 _resume.pth 都可）
        optimizer: 可选；提供时尝试恢复 optimizer state（仅 resume 文件有）
        scheduler: 可选；恢复 ``current_step``
        scaler:    可选；恢复 GradScaler state（仅 resume 文件有）
        device:    map_location
        strict:    state_dict load 严格模式

    Returns:
        ``{"step": int, "epoch": int, "loss": float}``
    """
    map_location = device if device else "cpu"
    checkpoint = torch.load(save_path, map_location=map_location, weights_only=False)

    # 兼容两种格式
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # 完整 resume 文件
        state_dict = checkpoint["model_state_dict"]
    else:
        # 纯权重文件（dict[str, Tensor]）或 OrderedDict
        state_dict = checkpoint
        checkpoint = {}  # 后续读取字段时按缺失处理

    # 取出真实 module（兼容 DDP / compile 包装）
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    # half 权重需要在加载时转回 model 的 dtype
    target_dtype = next(raw_model.parameters()).dtype
    if any(v.dtype == torch.float16 for v in state_dict.values() if isinstance(v, torch.Tensor)):
        state_dict = {k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    try:
        raw_model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        # 把"shape 对不上"的报错翻译成人话：通常是改了模型 config 之后还想 resume
        # 旧 ckpt（d_model / n_layers / vocab_size 任一变化都会触发）
        msg = str(e)
        if "size mismatch" in msg:
            # 用 token_embedding 推断 ckpt 的关键维度（vocab × d_model）
            ckpt_d_model = ckpt_vocab = "?"
            cur_d_model = cur_vocab = "?"
            ckpt_n_layers = cur_n_layers = "?"
            try:
                te = state_dict.get("token_embedding.weight")
                if te is not None and te.dim() == 2:
                    ckpt_vocab, ckpt_d_model = te.shape[0], te.shape[1]
                cur_te = raw_model.token_embedding.weight
                cur_vocab, cur_d_model = cur_te.shape[0], cur_te.shape[1]
                # n_layers：从 state_dict key 里找最大 layers.<idx>
                ckpt_layer_idx = [
                    int(k.split(".")[1]) for k in state_dict
                    if k.startswith("layers.") and k.split(".")[1].isdigit()
                ]
                ckpt_n_layers = (max(ckpt_layer_idx) + 1) if ckpt_layer_idx else "?"
                cur_n_layers = len(raw_model.layers)
            except Exception:
                pass

            stage_dir = os.path.dirname(save_path)
            raise RuntimeError(
                "\n"
                "❌ Checkpoint 与当前模型结构不兼容（shape mismatch）\n"
                f"   Checkpoint  : {save_path}\n"
                f"     vocab × d_model = {ckpt_vocab} × {ckpt_d_model}, n_layers = {ckpt_n_layers}\n"
                f"   当前 config :\n"
                f"     vocab × d_model = {cur_vocab} × {cur_d_model}, n_layers = {cur_n_layers}\n"
                "\n"
                "原因：模型 config 改了（d_model / n_layers / vocab_size 等任一变更），\n"
                "      旧 checkpoint 的权重 shape 与新模型对不上。\n"
                "\n"
                "解决方法（任选其一）：\n"
                f"  1) 续训没有意义 → 清空旧产物，从头训：\n"
                f"       rm -rf {stage_dir}\n"
                f"  2) 仅丢弃续训进度，保留历史 final.pth：\n"
                f"       rm {stage_dir}/_resume.pth\n"
                f"  3) 想继续用旧模型 → 把 config 改回旧值（d_model={ckpt_d_model}, n_layers={ckpt_n_layers}）\n"
            ) from e
        # 其它 RuntimeError 原样抛
        raise
    print(f"📦 模型权重加载: {save_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"   优化器状态恢复")

    if scheduler is not None and "scheduler_step" in checkpoint:
        scheduler.current_step = checkpoint["scheduler_step"]
        print(f"   调度器状态恢复 (step={scheduler.current_step})")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"   GradScaler 状态恢复")

    info = {
        "step": int(checkpoint.get("step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "loss": float(checkpoint.get("loss", 0.0)),
    }
    if checkpoint:
        print(f"   恢复到 step={info['step']}, epoch={info['epoch']}, loss={info['loss']:.4f}")
    return info


def find_resume_checkpoint(stage_dir: str) -> str | None:
    """在 stage 目录下查找 ``_resume.pth``（用于自动续训）

    优先级：
      1. ``<stage_dir>/_resume.pth`` 存在 → 返回它
      2. 否则返回 ``None``

    Args:
        stage_dir: 训练阶段输出目录，例如 ``outputs/pretrain``
    """
    resume_path = os.path.join(stage_dir, "_resume.pth")
    return resume_path if os.path.exists(resume_path) else None


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
        wandb_config:     可选 wandb / swanlab 配置 dict（None 关闭）。示例：
                          ``{"project": "ClearMind-Plus", "run_name": "...",
                            "backend": "swanlab", "wandb_id": "<resume_id>"}``
                          backend 取 "wandb" 或 "swanlab"（默认 swanlab，国内友好）
    """

    def __init__(
        self,
        log_dir: str = None,
        log_every: int = 10,
        use_tensorboard: bool = False,
        wandb_config: dict | None = None,
    ):
        self.log_every = log_every
        self.log_dir = log_dir
        self.history = []
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.tb_writer = None
        self.wandb_run = None  # wandb / swanlab Run 对象
        self._wandb_backend = None  # "wandb" / "swanlab" / None

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

        # wandb / swanlab（按需，缺依赖时优雅降级）
        if wandb_config:
            self._init_wandb(wandb_config)

    def _init_wandb(self, cfg: dict) -> None:
        """初始化 wandb / swanlab 后端

        cfg 关键字段：
          - backend: "wandb" / "swanlab"（默认 swanlab）
          - project: 项目名
          - run_name: run 名（默认自动生成）
          - wandb_id: 续训时用同一 id 让训练曲线连续
          - tags: list[str]
          - config: 透传给 wandb.init 的实验配置 dict
        """
        backend = cfg.get("backend", "swanlab").lower()
        project = cfg.get("project", "ClearMind")
        run_name = cfg.get("run_name")
        wandb_id = cfg.get("wandb_id")
        resume_mode = "must" if wandb_id else None

        try:
            if backend == "swanlab":
                import swanlab as W
            else:
                import wandb as W
        except ImportError as e:
            print(
                f"⚠️  {backend} 未安装，跳过远程监控（pip install {backend}）：{e}"
            )
            return

        try:
            run = W.init(
                project=project,
                name=run_name,
                id=wandb_id,
                resume=resume_mode,
                tags=cfg.get("tags"),
                config=cfg.get("config"),
            )
            self.wandb_run = run
            self._wandb_backend = backend
            run_id = getattr(run, "id", None) if run else None
            print(f"📈 {backend} 已启用，project={project}, run_id={run_id}")
        except Exception as e:
            print(f"⚠️  {backend}.init 失败: {e}")

    def get_wandb_id(self) -> str | None:
        """返回当前 wandb/swanlab run id（用于 _resume.pth 保存）"""
        if self.wandb_run is None:
            return None
        return getattr(self.wandb_run, "id", None)

    def _wandb_log(self, metrics: dict, step: int | None = None) -> None:
        """统一的 wandb / swanlab log 接口（缺时静默）"""
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.log(metrics, step=step)
        except Exception:
            pass

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

        # wandb / swanlab 写入（与 TensorBoard 并行）
        if self.wandb_run is not None:
            wandb_metrics = {"train/loss": loss, "train/lr": lr}
            if grad_norm > 0:
                wandb_metrics["train/grad_norm"] = grad_norm
            if "tokens_per_sec" in entry:
                wandb_metrics["train/tokens_per_sec"] = entry["tokens_per_sec"]
            self._wandb_log(wandb_metrics, step=step)

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
        if self.wandb_run is not None:
            self._wandb_log({"val/loss": val_loss}, step=step)

    def summary(self) -> None:
        """打印训练摘要"""
        if not self.history:
            return

        total_time = time.time() - self.start_time
        total_steps = len(self.history)
        final_loss = self.history[-1]["loss"]
        min_loss = min(e["loss"] for e in self.history)

        print("\n📊 训练摘要:")
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
        # 关闭 wandb / swanlab
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


# ============================================================
# SkipBatchSampler — 精确批跳过（长训续训不重过数据）
# ============================================================


class SkipBatchSampler:
    """跳过前 ``skip_batches`` 个 batch 的 BatchSampler

    用于 SFT/DPO/Distillation 等 epoch-based trainer 续训：
      - 旧实现：重启后从 epoch=0 / step=0 开始，前面已经过的数据要重新过一遍
      - 新实现：根据 ``_resume.pth`` 中保存的 step，精确跳过对应数量的 batch
        让续训直接从中断点的下一个 batch 开始

    参考 minimind/trainer/trainer_utils.py 的 SkipBatchSampler 实现。

    用法（在 trainer 的 train_loader 替换中）：

        from torch.utils.data import DataLoader
        from src.training.trainer_utils import SkipBatchSampler

        sampler = SkipBatchSampler(
            inner_sampler=base_sampler,           # 通常是 RandomSampler 或 DistributedSampler
            batch_size=32,
            skip_batches=resume_step,             # 从 _resume.pth 拿到的 step
        )
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, ...)

    Args:
        inner_sampler: 内部 sampler（可迭代对象，每次 yield 一个样本 idx）
        batch_size:    batch 大小
        skip_batches:  要跳过的 batch 数（>0 时启用，<=0 时不跳）
    """

    def __init__(
        self,
        inner_sampler,
        batch_size: int,
        skip_batches: int = 0,
    ):
        self.inner_sampler = inner_sampler
        self.batch_size = batch_size
        self.skip_batches = max(0, int(skip_batches))

    def __iter__(self):
        batch: list = []
        skipped = 0
        for idx in self.inner_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理 epoch 末尾不足一个 batch 的剩余
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:
        # 总 batch 数 = ceil(N / batch_size) - skip_batches
        try:
            n = len(self.inner_sampler)
        except TypeError:
            return 0  # 无限/不可知长度
        total_batches = (n + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)

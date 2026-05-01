"""
launch_ddp.py — DDP 多 GPU 预训练入口
=====================================

使用 ``torchrun`` + DistributedDataParallel 在多 GPU 上执行预训练。
默认走 minimind 数据 + HF tokenizer，配置选 main / plus 等已在 yaml 里写好的规格。

用法:

  # 单机 2 GPU
  torchrun --nproc_per_node=2 scripts/launch_ddp.py --config configs/main.yaml

  # 4 GPU + 覆盖参数
  torchrun --nproc_per_node=4 scripts/launch_ddp.py \\
      --config configs/plus.yaml \\
      --max_steps 50000 \\
      --batch_size 16

  # 断点续训
  torchrun --nproc_per_node=4 scripts/launch_ddp.py \\
      --config configs/plus.yaml \\
      --resume outputs/pretrain_ddp/checkpoint_step10000.pth
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _unwrap_model(model):
    """从 DDP / torch.compile 包装中取出原始模块"""
    inner = model.module if hasattr(model, "module") else model
    return getattr(inner, "_orig_mod", inner)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClearMind DDP 多 GPU 预训练")
    parser.add_argument(
        "--config", type=str, required=True, help="模型配置（如 configs/main.yaml）"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/pretrain_t2t_mini.jsonl",
        help="预训练数据路径（minimind 扁平格式）",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer 路径（默认读 yaml.tokenizer.path）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pretrain_ddp",
        help="输出目录",
    )
    parser.add_argument("--resume", type=str, default=None, help="续训 checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="覆盖 yaml.pretrain.max_steps")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 yaml.pretrain.batch_size")
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=None,
        help="覆盖 yaml.pretrain.gradient_accumulation",
    )
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")
    parser.add_argument("--backend", type=str, default="nccl", help="DDP backend")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # 延迟导入：避免 --help 时拉满依赖
    from src.model.gpt import GPT
    from src.model.config import ModelConfig
    from src.data.pretrain_dataset import PretrainDataset
    from src.training.trainer_utils import (
        setup_ddp,
        cleanup_ddp,
        is_main_process,
        wrap_model_ddp,
        get_dtype,
        CosineWarmupScheduler,
        clip_grad_norm,
        save_checkpoint,
        load_checkpoint,
        TrainingLogger,
        create_grad_scaler,
        amp_autocast,
    )

    # 复用 train.py 的 tokenizer 加载分支
    from scripts.train import load_tokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP 预训练需要 CUDA 环境。请在多 GPU 机器上使用 torchrun 启动。"
        )

    setup_ddp(local_rank, world_size, backend=args.backend)
    device = torch.device(f"cuda:{local_rank}")

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        if "model" not in config or "pretrain" not in config:
            raise ValueError(
                f"配置文件缺少 `model` 或 `pretrain` 字段: {args.config}"
            )

        # ---- pretrain 配置 + CLI 覆盖 ----
        train_cfg = dict(config["pretrain"])
        for cli_key, yaml_key in [
            ("max_steps", "max_steps"),
            ("batch_size", "batch_size"),
            ("gradient_accumulation", "gradient_accumulation"),
            ("save_every", "save_every"),
            ("log_every", "log_every"),
        ]:
            v = getattr(args, cli_key)
            if v is not None:
                train_cfg[yaml_key] = v

        if is_main_process():
            print(f"🚀 DDP 启动: world_size={world_size}")
            print(f"📄 Config:    {args.config}")
            print(f"📦 Data:      {args.data}")
            print(f"📁 Output:    {args.output_dir}")

        if not os.path.exists(args.data):
            raise FileNotFoundError(
                f"预训练数据不存在: {args.data}\n"
                "请从 modelscope/HF 下载 pretrain_t2t_mini.jsonl 放入 data/"
            )

        # ---- Tokenizer + Model ----
        tokenizer = load_tokenizer(config, args.tokenizer)
        model_config = ModelConfig(**config["model"])
        if tokenizer.vocab_size != model_config.vocab_size:
            if is_main_process():
                print(
                    f"⚠️ tokenizer.vocab_size ({tokenizer.vocab_size}) ≠ "
                    f"model.vocab_size ({model_config.vocab_size})，自动对齐"
                )
            model_config.vocab_size = tokenizer.vocab_size

        # ---- Dataset ----
        pretrain_mode = train_cfg.get("pretrain_mode", "per_sample")
        dataset = PretrainDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
            mode=pretrain_mode,
        )
        if len(dataset) == 0:
            raise ValueError("预训练数据样本数为 0，请检查数据质量或增大数据量")

        batch_size = int(train_cfg.get("batch_size", 8))
        grad_accum = int(train_cfg.get("gradient_accumulation", 2))
        max_steps = int(train_cfg.get("max_steps", 10000))
        save_every = int(train_cfg.get("save_every", 1000))
        log_every = int(train_cfg.get("log_every", 50))

        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        if sampler.num_samples == 0 or len(loader) == 0:
            raise ValueError(
                "每个 rank 可用样本不足。请减小 batch_size / 增大数据量"
            )

        # ---- Optimizer + Scheduler + Scaler ----
        dtype = get_dtype(device, train_cfg.get("dtype", "bfloat16"))
        model = GPT(model_config).to(device)
        model = wrap_model_ddp(model, device_id=local_rank)

        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": train_cfg.get("weight_decay", 0.01),
                },
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.get("lr", 3e-4),
            betas=(0.9, 0.95),
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_lr=train_cfg.get("lr", 3e-4),
            min_lr=train_cfg.get("min_lr", train_cfg.get("lr", 3e-4) * 0.1),
            warmup_steps=train_cfg.get("warmup_steps", 1000),
            max_steps=max_steps,
        )
        scaler = create_grad_scaler(device, dtype)

        # ---- Logger ----
        os.makedirs(args.output_dir, exist_ok=True)
        logger = None
        if is_main_process():
            logger = TrainingLogger(
                log_dir=os.path.join(args.output_dir, "logs"),
                log_every=log_every,
                use_tensorboard=train_cfg.get("use_tensorboard", False),
            )
            params_m = _unwrap_model(model).count_parameters()["total_millions"]
            print(f"🧠 模型参数量: {params_m:.1f}M")
            print(
                f"📋 batch_size={batch_size}, grad_accum={grad_accum}, "
                f"effective_batch={batch_size * grad_accum * world_size}, "
                f"max_steps={max_steps}, dtype={dtype}"
            )

        # ---- Resume ----
        step = 0
        if args.resume:
            info = load_checkpoint(
                _unwrap_model(model),
                args.resume,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            step = int(info["step"])
            if is_main_process():
                print(f"🔄 从 step={step} 恢复: {args.resume}")
        if step >= max_steps:
            if is_main_process():
                print("✅ checkpoint 已达 max_steps，无需继续训练")
            return

        # ---- Train Loop ----
        model.train()
        optimizer.zero_grad()
        data_iter = iter(loader)
        sampler_epoch = 0
        sampler.set_epoch(sampler_epoch)

        while step < max_steps:
            micro_loss_sum = 0.0
            tokens_per_step = 0

            # ---- DDP no_sync 优化（Phase 4）----
            # 梯度累积时，前 grad_accum-1 个 micro-batch 用 no_sync() 跳过 all-reduce，
            # 最后一个 micro-batch 才同步。能节省 ~30% 通信带宽（grad_accum 越大越显著）。
            # 对 single-GPU（world_size=1）no_sync 是 noop，不影响。
            from contextlib import nullcontext
            is_ddp = hasattr(model, "no_sync")

            for micro_idx in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    sampler_epoch += 1
                    sampler.set_epoch(sampler_epoch)
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                tokens_per_step += input_ids.numel() * world_size

                # 最后一个 micro-batch 才允许同步梯度
                is_last_micro = (micro_idx == grad_accum - 1)
                sync_ctx = nullcontext() if (is_last_micro or not is_ddp) else model.no_sync()

                with sync_ctx:
                    with amp_autocast(device, dtype):
                        _, loss, _ = model(input_ids, labels)

                    scaled_loss = loss / grad_accum
                    if scaler:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                micro_loss_sum += loss.item()

            if scaler:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm(model, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = clip_grad_norm(model, max_norm=1.0)
                optimizer.step()

            lr = scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            avg_loss = micro_loss_sum / grad_accum
            step += 1

            if is_main_process() and logger:
                logger.log(
                    step=step,
                    max_steps=max_steps,
                    loss=avg_loss,
                    lr=lr,
                    tokens_per_step=tokens_per_step,
                    grad_norm=grad_norm,
                )
                if step % save_every == 0:
                    save_checkpoint(
                        model=_unwrap_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        loss=avg_loss,
                        save_path=os.path.join(
                            args.output_dir, f"checkpoint_step{step}.pth"
                        ),
                    )

        # ---- 保存 final ----
        if is_main_process():
            save_checkpoint(
                model=_unwrap_model(model),
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=avg_loss,
                save_path=os.path.join(args.output_dir, "final.pth"),
            )
            if logger:
                logger.save_log("ddp_pretrain_log.jsonl")
                logger.summary()
            print("✅ DDP 预训练完成")

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

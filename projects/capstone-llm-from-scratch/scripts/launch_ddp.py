"""
launch_ddp.py — DDP 多 GPU 预训练启动脚本
=========================================

使用 torchrun + DistributedDataParallel 在多 GPU 上执行预训练。

用法:
  # 单机 2 GPU
  torchrun --nproc_per_node=2 scripts/launch_ddp.py --config configs/medium.yaml

  # 覆盖参数
  torchrun --nproc_per_node=4 scripts/launch_ddp.py \
      --config configs/large.yaml \
      --max_steps 2000 \
      --batch_size 4 \
      --data data/pretrain/pretrain_data.jsonl
"""

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
    """DDP 包装模型取出原始模块。"""
    return model.module if hasattr(model, "module") else model


def _build_parser():
    parser = argparse.ArgumentParser(description="DDP 多 GPU 预训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--data",
        type=str,
        default="data/pretrain/pretrain_data.jsonl",
        help="预训练数据路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="outputs/tokenizer/tokenizer.model",
        help="分词器路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pretrain_ddp",
        help="输出目录",
    )
    parser.add_argument("--resume", type=str, default=None, help="恢复训练 checkpoint")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="覆盖配置中的 pretrain.max_steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="覆盖配置中的 pretrain.batch_size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=None,
        help="覆盖配置中的 pretrain.gradient_accumulation",
    )
    parser.add_argument(
        "--save_every", type=int, default=None, help="覆盖配置中的 pretrain.save_every"
    )
    parser.add_argument(
        "--log_every", type=int, default=None, help="覆盖配置中的 pretrain.log_every"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="DataLoader worker 数"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="DDP backend (默认 nccl)"
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # 延迟导入, 保证 --help 不受可选依赖影响
    from src.model.gpt import GPT
    from src.model.config import ModelConfig
    from src.data.pretrain_dataset import PretrainDataset
    from src.data.tokenizer import ClearMindTokenizer
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
            raise ValueError(f"配置文件缺少 `model` 或 `pretrain` 字段: {args.config}")

        train_cfg = dict(config["pretrain"])
        if args.max_steps is not None:
            train_cfg["max_steps"] = args.max_steps
        if args.batch_size is not None:
            train_cfg["batch_size"] = args.batch_size
        if args.gradient_accumulation is not None:
            train_cfg["gradient_accumulation"] = args.gradient_accumulation
        if args.save_every is not None:
            train_cfg["save_every"] = args.save_every
        if args.log_every is not None:
            train_cfg["log_every"] = args.log_every

        if is_main_process():
            print(f"🚀 DDP 启动: {world_size} GPU(s)")
            print(f"📄 配置: {args.config}")
            print(f"📦 数据: {args.data}")
            print(f"🔤 分词器: {args.tokenizer}")
            print(f"📁 输出目录: {args.output_dir}")

        if not os.path.exists(args.data):
            raise FileNotFoundError(f"预训练数据不存在: {args.data}")
        if not os.path.exists(args.tokenizer):
            raise FileNotFoundError(f"分词器不存在: {args.tokenizer}")

        tokenizer = ClearMindTokenizer(args.tokenizer)
        model_config = ModelConfig(**config["model"])
        if tokenizer.vocab_size != model_config.vocab_size:
            if is_main_process():
                print(
                    f"⚠️  vocab_size 不一致，自动更新: "
                    f"{model_config.vocab_size} → {tokenizer.vocab_size}"
                )
            model_config.vocab_size = tokenizer.vocab_size

        dataset = PretrainDataset(
            data_path=args.data,
            tokenizer=tokenizer,
            max_seq_len=model_config.max_seq_len,
        )
        if len(dataset) == 0:
            raise ValueError("预训练数据样本数为 0，请检查数据质量或增大数据量")

        batch_size = int(train_cfg.get("batch_size", 8))
        grad_accum = int(train_cfg.get("gradient_accumulation", 2))
        max_steps = int(train_cfg.get("max_steps", 10000))
        save_every = int(train_cfg.get("save_every", 1000))
        log_every = int(train_cfg.get("log_every", 10))

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
        )

        if sampler.num_samples == 0 or len(loader) == 0:
            raise ValueError(
                "每个 rank 可用样本不足，无法构建 DDP DataLoader。"
                "请减小 batch_size / 增大数据量。"
            )

        dtype = get_dtype(device, train_cfg.get("dtype", "float32"))
        model = GPT(model_config).to(device)
        model = wrap_model_ddp(model, device_id=local_rank)

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": train_cfg.get("weight_decay", 0.01)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=train_cfg.get("lr", 3e-4), betas=(0.9, 0.95))
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_lr=train_cfg.get("lr", 3e-4),
            min_lr=train_cfg.get("min_lr", train_cfg.get("lr", 3e-4) * 0.1),
            warmup_steps=train_cfg.get("warmup_steps", 500),
            max_steps=max_steps,
        )
        scaler = create_grad_scaler(device, dtype)

        os.makedirs(args.output_dir, exist_ok=True)
        logger = None
        if is_main_process():
            logger = TrainingLogger(
                log_dir=os.path.join(args.output_dir, "logs"),
                log_every=log_every,
                use_tensorboard=train_cfg.get("use_tensorboard", False),
            )
            total_params = _unwrap_model(model).count_parameters()["total_millions"]
            print(f"🧠 模型参数量: {total_params:.1f}M")
            print(
                f"📋 训练配置: batch_size={batch_size}, grad_accum={grad_accum}, "
                f"max_steps={max_steps}, dtype={dtype}"
            )

        step = 0
        if args.resume:
            resume_info = load_checkpoint(
                _unwrap_model(model),
                args.resume,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            step = int(resume_info["step"])
            if is_main_process():
                print(f"🔄 从 step={step} 恢复训练: {args.resume}")

        if step >= max_steps:
            if is_main_process():
                print("✅ checkpoint 已达到 max_steps，无需继续训练")
            return

        model.train()
        optimizer.zero_grad()
        data_iter = iter(loader)
        sampler_epoch = 0
        sampler.set_epoch(sampler_epoch)

        while step < max_steps:
            micro_loss_sum = 0.0
            tokens_per_step = 0

            for _ in range(grad_accum):
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
            optimizer.zero_grad()

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

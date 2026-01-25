"""
训练脚本
========

实现完整的 LoRA 微调训练流程。

学习要点：
    1. 训练循环的标准实现
    2. 梯度累积提高有效批次大小
    3. 学习率调度器的使用
    4. 验证和检查点保存
"""

import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config
from model import create_model, TinyLLM
from dataset import create_dataloaders, SimpleTokenizer
from utils import (
    set_seed,
    get_device,
    save_checkpoint,
    save_lora_weights,
    AverageMeter,
    EarlyStopping,
    print_model_info,
)


def train_epoch(
    model: TinyLLM,
    train_loader,
    optimizer,
    scheduler,
    device: torch.device,
    config: Config,
    epoch: int,
) -> float:
    """训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        config: 配置
        epoch: 当前轮数

    Returns:
        平均训练损失
    """
    model.train()
    loss_meter = AverageMeter()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # 移动到设备
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=None, labels=labels)
        loss = outputs["loss"]

        # 梯度累积
        loss = loss / config.training.gradient_accumulation_steps
        loss.backward()

        loss_meter.update(loss.item() * config.training.gradient_accumulation_steps)

        # 梯度更新
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 日志
        if (batch_idx + 1) % config.training.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch + 1} | "
                f"Step {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss_meter.avg:.4f} | "
                f"LR: {lr:.2e}"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(model: TinyLLM, val_loader, device: torch.device) -> float:
    """验证模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 计算设备

    Returns:
        平均验证损失
    """
    model.eval()
    loss_meter = AverageMeter()

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss_meter.update(outputs["loss"].item())

    return loss_meter.avg


def train(config: Optional[Config] = None, num_epochs: Optional[int] = None):
    """主训练函数

    Args:
        config: 配置对象，如果为 None 则使用默认配置
        num_epochs: 训练轮数，覆盖配置中的值
    """
    if config is None:
        config = get_config()

    if num_epochs is not None:
        config.training.num_epochs = num_epochs

    # 设置随机种子
    set_seed(config.training.seed)

    # 获取设备
    device = get_device()
    print(f"\n使用设备: {device}")

    # 创建分词器
    tokenizer = SimpleTokenizer(config.model.vocab_size)

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        data_config=config.data,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length,
    )
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")

    # 创建模型
    model = create_model(config.model, use_lora=True, lora_config=config.lora)
    model = model.to(device)
    print_model_info(model, "TinyLLM + LoRA")

    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # 学习率调度器
    total_steps = (
        len(train_loader)
        * config.training.num_epochs
        // config.training.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config.training.learning_rate * 0.1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # 早停
    early_stopping = EarlyStopping(patience=5)

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(config.training.num_epochs):
        epoch_start = time.time()

        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch
        )

        # 验证
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_lora_weights(model, os.path.join(config.model_dir, "best_lora.pt"))

        # 早停检查
        if early_stopping(val_loss):
            print("\n早停触发！")
            break

    total_time = time.time() - start_time

    # 保存最终模型
    save_checkpoint(
        model,
        optimizer,
        epoch,
        val_loss,
        os.path.join(config.model_dir, "final_checkpoint.pt"),
    )
    save_lora_weights(model, os.path.join(config.model_dir, "final_lora.pt"))

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"总时间: {total_time:.1f}s")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存于: {config.model_dir}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练 LLM")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    args = parser.parse_args()

    config = get_config()
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    train(config)

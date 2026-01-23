"""
训练器 - 核心训练逻辑
包含所有 Phase 9 优化技巧
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:
    """
    训练器类

    包含的优化技巧:
    - 多种优化器 (SGD, Adam, AdamW)
    - 学习率调度 (Step, Cosine, OneCycle)
    - 学习率预热 (Warmup)
    - 混合精度训练 (AMP)
    - 梯度累积
    - 梯度裁剪
    """

    def __init__(self, model, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 设备
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = model.to(self.device)
        print(f"使用设备: {self.device}")

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "lr": [],
        }

        # 最佳模型
        self.best_acc = 0.0

    def _create_optimizer(self):
        """创建优化器"""
        config = self.config

        if config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"未知优化器: {config.optimizer}")

        print(f"优化器: {config.optimizer}, lr={config.lr}")
        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        config = self.config
        total_steps = len(self.train_loader) * config.epochs

        if config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.min_lr
            )
        elif config.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.lr,
                total_steps=total_steps,
                pct_start=0.3,
            )
        elif config.scheduler == "none":
            scheduler = None
        else:
            raise ValueError(f"未知调度器: {config.scheduler}")

        print(f"调度器: {config.scheduler}")
        return scheduler

    def _warmup_lr(self, epoch, batch_idx):
        """学习率预热"""
        if epoch < self.config.warmup_epochs:
            warmup_steps = self.config.warmup_epochs * len(self.train_loader)
            current_step = epoch * len(self.train_loader) + batch_idx
            warmup_factor = current_step / warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config.lr * warmup_factor

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 学习率预热
            if self.config.warmup_epochs > 0:
                self._warmup_lr(epoch, batch_idx)

            # 混合精度前向传播
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.gradient_accumulation_steps

            # 反向传播
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.grad_clip is not None:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                # 优化器更新
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # OneCycle 每步更新
                if self.config.scheduler == "onecycle" and self.scheduler:
                    self.scheduler.step()

            # 统计
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": total_loss / (batch_idx + 1),
                    "acc": 100.0 * correct / total,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        # 非 OneCycle 的调度器在 epoch 结束后更新
        if self.config.scheduler != "onecycle" and self.scheduler:
            self.scheduler.step()

        train_loss = total_loss / len(self.train_loader)
        train_acc = 100.0 * correct / total

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_loss = total_loss / len(self.test_loader)
        test_acc = 100.0 * correct / total

        return test_loss, test_acc

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 评估
            test_loss, test_acc = self.evaluate()

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(
                f"\nEpoch {epoch + 1}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
            )

            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_model("best_model.pth")
                print(f"  → 新最佳模型! 准确率: {test_acc:.2f}%")

            # 定期保存
            if (epoch + 1) % self.config.save_every == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"训练完成! 总时间: {total_time / 60:.2f} 分钟")
        print(f"最佳测试准确率: {self.best_acc:.2f}%")
        print("=" * 60)

        return self.history

    def save_model(self, filename):
        """保存模型"""
        os.makedirs(f"{self.config.output_dir}/models", exist_ok=True)
        path = f"{self.config.output_dir}/models/{filename}"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "config": self.config,
            },
            path,
        )

    def load_model(self, filename):
        """加载模型"""
        path = f"{self.config.output_dir}/models/{filename}"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_acc = checkpoint["best_acc"]
        print(f"加载模型: {path}, 准确率: {self.best_acc:.2f}%")

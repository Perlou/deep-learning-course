"""训练工具函数单元测试"""

import torch
import torch.nn as nn
import pytest

from training.trainer_utils import (
    CosineWarmupScheduler,
    EarlyStopping,
    clip_grad_norm,
    evaluate_loss,
    create_grad_scaler,
)


class TestCosineWarmupScheduler:
    """测试余弦退火 + 线性预热调度器"""

    def setup_method(self):
        """每个测试方法前创建 scheduler"""
        self.model = nn.Linear(10, 10)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=100,
            max_steps=1000,
        )

    def test_warmup_starts_near_zero(self):
        """warmup 开始时 LR 应接近 0"""
        lr = self.scheduler.get_lr(0)
        assert lr < 1e-4

    def test_warmup_end(self):
        """warmup 结束时 LR 应达到 max_lr"""
        lr = self.scheduler.get_lr(100)
        assert abs(lr - 1e-3) < 1e-6

    def test_cosine_decay(self):
        """cosine 阶段 LR 应逐渐降低"""
        lr_200 = self.scheduler.get_lr(200)
        lr_500 = self.scheduler.get_lr(500)
        lr_900 = self.scheduler.get_lr(900)
        assert lr_200 > lr_500 > lr_900

    def test_end_lr(self):
        """训练结束时 LR 应接近 min_lr"""
        lr = self.scheduler.get_lr(1000)
        assert abs(lr - 1e-5) < 1e-5

    def test_step_updates(self):
        """step() 应更新 optimizer 中的 LR"""
        for _ in range(100):
            self.scheduler.step()
        for pg in self.optimizer.param_groups:
            assert abs(pg["lr"] - 1e-3) < 1e-4, (
                f"warmup 结束后 LR 应接近 max_lr, 实际: {pg['lr']}"
            )


class TestEarlyStopping:
    """测试 Early Stopping"""

    def test_no_stop_when_improving(self):
        """指标持续改善时不应停止"""
        es = EarlyStopping(patience=3, mode="min")
        for val in [10, 9, 8, 7, 6]:
            assert es(val) is False

    def test_stop_after_patience(self):
        """指标连续 patience 次不改善时应停止"""
        es = EarlyStopping(patience=3, mode="min")
        es(1.0)  # best
        es(1.1)  # worse, counter=1
        es(1.2)  # worse, counter=2
        result = es(1.3)  # worse, counter=3 → stop
        assert result is True
        assert es.should_stop is True

    def test_max_mode(self):
        """max 模式: 值越大越好"""
        es = EarlyStopping(patience=2, mode="max")
        es(0.5)  # best
        assert es(0.6) is False  # improved
        assert es(0.4) is False  # worse, counter=1
        assert es(0.3) is True  # worse, counter=2 → stop

    def test_min_delta(self):
        """min_delta: 微小改善不计入改善"""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        es(1.0)  # best
        assert es(0.95) is False  # improved but < delta, counter=1
        assert es(0.96) is True  # counter=2 → stop

    def test_best_score_tracking(self):
        """best_score 应实时跟踪"""
        es = EarlyStopping(patience=5, mode="min")
        es(10)
        assert es.best_score == 10
        es(8)
        assert es.best_score == 8
        es(9)
        assert es.best_score == 8  # 未改善, best 不变


class TestClipGradNorm:
    """测试梯度裁剪"""

    def test_clip(self):
        """裁剪应返回裁剪前的梯度范数"""
        model = nn.Linear(10, 10)
        x = torch.randn(1, 10)
        y = model(x).sum()
        y.backward()

        grad_norm = clip_grad_norm(model, max_norm=1.0)
        assert grad_norm >= 0

    def test_grad_norm_capped(self):
        """裁剪后梯度范数应 <= max_norm"""
        model = nn.Linear(10, 10)
        # 人为制造大梯度
        x = torch.randn(1, 10) * 100
        y = model(x).sum()
        y.backward()

        clip_grad_norm(model, max_norm=1.0)

        # 验证裁剪后的范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= 1.0 + 1e-6


class TestGradScaler:
    """测试 GradScaler 创建"""

    def test_cpu_returns_none(self):
        """CPU 设备应返回 None"""
        scaler = create_grad_scaler(torch.device("cpu"), torch.float32)
        assert scaler is None

    def test_cpu_fp16_returns_none(self):
        """CPU + fp16 也应返回 None"""
        scaler = create_grad_scaler(torch.device("cpu"), torch.float16)
        assert scaler is None

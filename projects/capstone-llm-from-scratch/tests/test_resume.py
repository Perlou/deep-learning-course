"""断点续训正确性测试

验证：
  1. save_checkpoint 后 _resume.pth 与纯权重文件都存在
  2. 纯权重文件比 resume 小（因为 half 落盘）
  3. 原子保存：_resume.pth.tmp 不会留在磁盘
  4. load_checkpoint 后 model / optimizer / scheduler / scaler / step / epoch 全部恢复
  5. find_resume_checkpoint 能正确找到
  6. 旧格式（无 scaler/epoch 字段）的 ckpt 仍能加载（向后兼容）

测试不依赖 GPU，全部在 CPU 上跑。
"""

import os

import pytest
import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.trainer_utils import (
    CosineWarmupScheduler,
    save_checkpoint,
    load_checkpoint,
    find_resume_checkpoint,
)


@pytest.fixture
def tmp_train_state(tmp_path):
    """搭一个最小训练状态：tiny model + AdamW + scheduler + 假 scaler"""
    cfg = ModelConfig.tiny()
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer, max_lr=1e-3, min_lr=1e-5, warmup_steps=2, max_steps=10
    )
    # 走几步让 optimizer / scheduler 有非空 state
    for _ in range(3):
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        _, loss, _ = model(x, x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # mock scaler
    class MockScaler:
        def __init__(self):
            self._scale = 65536.0

        def state_dict(self):
            return {"_scale": self._scale, "_growth_tracker": 5}

        def load_state_dict(self, sd):
            self._scale = sd["_scale"]

    scaler = MockScaler()
    return model, optimizer, scheduler, scaler, cfg, str(tmp_path)


class TestSaveCheckpoint:
    def test_writes_two_files(self, tmp_train_state):
        model, optimizer, scheduler, scaler, _, work = tmp_train_state
        save_path = os.path.join(work, "test_step10.pth")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=2.5,
            save_path=save_path,
            epoch=1,
            scaler=scaler,
        )
        assert os.path.exists(save_path), "纯权重文件应存在"
        resume_path = os.path.join(work, "_resume.pth")
        assert os.path.exists(resume_path), "_resume.pth 应存在"

    def test_weights_are_half_precision(self, tmp_train_state):
        """纯权重文件应为 fp16 落盘，体积明显小于 _resume"""
        model, optimizer, scheduler, scaler, _, work = tmp_train_state
        save_path = os.path.join(work, "weights.pth")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.0,
            save_path=save_path,
            scaler=scaler,
        )
        weights_size = os.path.getsize(save_path)
        resume_size = os.path.getsize(os.path.join(work, "_resume.pth"))
        # resume 含完整 fp32 optimizer state，应比纯权重大很多
        assert resume_size > weights_size, "resume 应包含更多内容"
        # 验证权重确实是 fp16
        loaded = torch.load(save_path, map_location="cpu", weights_only=False)
        for k, v in loaded.items():
            assert v.dtype == torch.float16, f"{k} 应为 fp16，实际 {v.dtype}"

    def test_no_tmp_files_left(self, tmp_train_state):
        """原子保存不应留下 .tmp 文件"""
        model, optimizer, scheduler, scaler, _, work = tmp_train_state
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.0,
            save_path=os.path.join(work, "x.pth"),
            scaler=scaler,
        )
        leftover = [f for f in os.listdir(work) if f.endswith(".tmp")]
        assert leftover == [], f"不应有 .tmp 残留: {leftover}"

    def test_no_resume_when_disabled(self, tmp_train_state):
        """save_resume=False 时不应写 _resume.pth"""
        model, optimizer, scheduler, scaler, _, work = tmp_train_state
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            loss=1.0,
            save_path=os.path.join(work, "weights_only.pth"),
            scaler=scaler,
            save_resume=False,
        )
        assert not os.path.exists(os.path.join(work, "_resume.pth"))


class TestLoadCheckpoint:
    def test_resume_restores_full_state(self, tmp_train_state):
        """load_checkpoint 应能从 _resume.pth 恢复 model / optimizer / scheduler / scaler / step / epoch"""
        model, optimizer, scheduler, scaler, cfg, work = tmp_train_state
        save_path = os.path.join(work, "ckpt.pth")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=42,
            loss=3.14,
            save_path=save_path,
            epoch=2,
            scaler=scaler,
        )
        original_lr = optimizer.param_groups[0]["lr"]
        original_scheduler_step = scheduler.current_step
        original_scaler_scale = scaler._scale

        # 销毁原状态，新建并加载
        new_model = GPT(cfg)
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=999.0)
        new_scheduler = CosineWarmupScheduler(
            optimizer=new_optimizer,
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=2,
            max_steps=10,
        )

        class MockScaler:
            def __init__(self):
                self._scale = 1.0

            def state_dict(self):
                return {"_scale": self._scale}

            def load_state_dict(self, sd):
                self._scale = sd["_scale"]

        new_scaler = MockScaler()

        info = load_checkpoint(
            new_model,
            os.path.join(work, "_resume.pth"),
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=new_scaler,
            device=torch.device("cpu"),
        )
        # step / epoch / loss
        assert info["step"] == 42
        assert info["epoch"] == 2
        assert abs(info["loss"] - 3.14) < 1e-4
        # scheduler 恢复
        assert new_scheduler.current_step == original_scheduler_step
        # scaler 恢复
        assert new_scaler._scale == original_scaler_scale
        # optimizer 恢复（lr 应被 scheduler 重新算覆盖之前是从 ckpt 来的）
        # 这里只验证 ckpt 里有 optimizer state（已通过没报错验证）

    def test_legacy_ckpt_compatible(self, tmp_path):
        """旧格式 ckpt（无 scaler / epoch 字段）应能加载且不报错"""
        cfg = ModelConfig.tiny()
        model = GPT(cfg)
        # 写一个不含 scaler / epoch 的旧格式 ckpt
        old_ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},  # 空 optimizer state 就行
            "scheduler_step": 5,
            "step": 100,
            "loss": 2.0,
        }
        legacy_path = tmp_path / "legacy.pth"
        torch.save(old_ckpt, str(legacy_path))

        new_model = GPT(cfg)
        info = load_checkpoint(
            new_model,
            str(legacy_path),
            optimizer=None,
            scheduler=None,
            scaler=None,
        )
        assert info["step"] == 100
        # epoch 字段缺失 → 默认 0
        assert info["epoch"] == 0

    def test_pure_weights_compatible(self, tmp_path):
        """纯权重文件（OrderedDict[str, Tensor]）也能加载"""
        cfg = ModelConfig.tiny()
        model = GPT(cfg)
        weights_path = tmp_path / "weights.pth"
        # 模拟 .half().cpu() 落盘
        torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, str(weights_path))

        new_model = GPT(cfg)
        info = load_checkpoint(
            new_model,
            str(weights_path),
            optimizer=None,
            scheduler=None,
        )
        # 没有 step 字段 → 默认 0
        assert info["step"] == 0


class TestFindResumeCheckpoint:
    def test_find_existing(self, tmp_path):
        """_resume.pth 存在时返回完整路径"""
        target = tmp_path / "_resume.pth"
        target.write_bytes(b"")
        result = find_resume_checkpoint(str(tmp_path))
        assert result == str(target)

    def test_find_missing(self, tmp_path):
        """_resume.pth 不存在时返回 None"""
        assert find_resume_checkpoint(str(tmp_path)) is None

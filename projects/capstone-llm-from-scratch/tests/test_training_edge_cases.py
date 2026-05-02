"""训练边界条件与包导入行为测试"""

import importlib
import importlib.util

import pytest
import torch
from torch.utils.data import Dataset


class TinyPretrainDataset(Dataset):
    """最小预训练数据集，用于边界条件测试。"""

    def __init__(self, seq_len: int = 16, vocab_size: int = 128):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }


class TinySFTDataset(Dataset):
    """最小 SFT 数据集，长度可配置。"""

    def __init__(self, size: int = 1, seq_len: int = 16, vocab_size: int = 128):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        labels = tokens.clone()
        labels[0] = -100
        return {
            "input_ids": tokens,
            "labels": labels,
        }


class TinyDPODataset(Dataset):
    """最小 DPO 数据集，长度可配置。"""

    def __init__(self, size: int = 1, seq_len: int = 16, vocab_size: int = 128):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        chosen = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        rejected = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)

        chosen_labels = chosen.clone()
        rejected_labels = rejected.clone()
        chosen_labels[0] = -100
        rejected_labels[0] = -100

        return {
            "chosen_input_ids": chosen,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected,
            "rejected_labels": rejected_labels,
        }


def test_data_package_lazy_tokenizer_import():
    """导入 data 包不应在无 sentencepiece 时阻断 dataset 使用。"""
    data_pkg = importlib.import_module("data")
    assert hasattr(data_pkg, "PretrainDataset")

    has_sentencepiece = importlib.util.find_spec("sentencepiece") is not None
    if has_sentencepiece:
        assert data_pkg.ClearMindTokenizer is not None
    else:
        with pytest.raises(ModuleNotFoundError):
            _ = data_pkg.ClearMindTokenizer


def test_pretrainer_handles_zero_max_steps(tiny_model, tmp_path):
    """max_steps=0 时应能优雅结束并保存 final checkpoint。"""
    from training.pretrain import PreTrainer

    dataset = TinyPretrainDataset(
        seq_len=16,
        vocab_size=tiny_model.config.vocab_size,
    )
    output_dir = tmp_path / "pretrain_zero_steps"

    trainer = PreTrainer(
        model=tiny_model,
        train_dataset=dataset,
        config={
            "max_steps": 0,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "log_every": 1,
            "save_every": 1,
        },
        output_dir=str(output_dir),
    )
    trainer.train()

    assert (output_dir / "final.pth").exists()


def test_epoch_trainers_use_ceil_steps_per_epoch(tiny_model, tmp_path):
    """SFT/DPO 在 len(loader) < grad_accum 时，steps_per_epoch 不应为 0。"""
    from training.sft import SFTTrainer
    from training.dpo import DPOTrainer

    sft_trainer = SFTTrainer(
        model=tiny_model,
        train_dataset=TinySFTDataset(
            size=1,
            seq_len=16,
            vocab_size=tiny_model.config.vocab_size,
        ),
        config={"epochs": 2, "batch_size": 1, "gradient_accumulation": 4},
        output_dir=str(tmp_path / "sft_steps"),
    )
    assert sft_trainer.steps_per_epoch == 1
    assert sft_trainer.max_steps == 2

    dpo_trainer = DPOTrainer(
        model=tiny_model,
        train_dataset=TinyDPODataset(
            size=1,
            seq_len=16,
            vocab_size=tiny_model.config.vocab_size,
        ),
        config={"epochs": 2, "batch_size": 1, "gradient_accumulation": 4},
        output_dir=str(tmp_path / "dpo_steps"),
    )
    assert dpo_trainer.steps_per_epoch == 1
    assert dpo_trainer.max_steps == 2


def test_dpo_trainer_respects_config_max_steps(tiny_model, tmp_path):
    """DPOTrainer 应允许 config.max_steps 下调总步数（与 SFTTrainer 行为对齐）。

    回归测试：旧实现 self.max_steps = steps_per_epoch * epochs 完全忽略
    config["max_steps"]，导致 yaml 里的 max_steps 字段失效，AutoDL 上跑全量
    DPO 时无法限制时长。
    """
    from training.dpo import DPOTrainer

    # 数据集大到能被 max_steps 截短：size=64, batch=1, grad_accum=1, epochs=2
    # → steps_per_epoch=64, full_max=128；用 max_steps=10 应该被尊重
    dpo_trainer = DPOTrainer(
        model=tiny_model,
        train_dataset=TinyDPODataset(
            size=64,
            seq_len=16,
            vocab_size=tiny_model.config.vocab_size,
        ),
        config={
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "max_steps": 10,
        },
        output_dir=str(tmp_path / "dpo_max_steps"),
    )
    assert dpo_trainer.steps_per_epoch == 64
    # 关键断言：max_steps 应该被 config 覆盖到 10，而不是 128
    assert dpo_trainer.max_steps == 10, (
        f"DPOTrainer 没有尊重 config.max_steps；"
        f"期望 10，实际 {dpo_trainer.max_steps}"
    )

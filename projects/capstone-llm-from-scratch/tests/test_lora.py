"""LoRA 单元测试"""

import torch
import torch.nn as nn
import pytest

from model import ModelConfig, GPT
from training.lora import (
    LoRALinear,
    apply_lora,
    merge_lora,
    lora_state_dict,
    load_lora_state_dict,
)


class TestLoRALinear:
    """测试 LoRALinear 层"""

    def test_output_shape(self):
        """LoRA 层输出维度应与原始 Linear 一致"""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_initial_output_matches(self):
        """初始化时 LoRA 修正应为 0 (B 初始化为 0)"""
        linear = nn.Linear(64, 128)
        x = torch.randn(2, 10, 64)

        # 原始输出
        orig_out = linear(x)

        # LoRA 层 (B=0 → ΔW=0)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        lora_out = lora(x)

        assert torch.allclose(orig_out, lora_out, atol=1e-6), "初始时 LoRA 不应改变输出"

    def test_lora_params_count(self):
        """LoRA 参数量应为 rank * (in + out)"""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4)
        assert lora.lora_params == 4 * (64 + 128)

    def test_merge(self):
        """merge 后输出应与 merge 前一致"""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            before_merge = lora(x).clone()

        lora.merge()

        # merge 后直接用原始 weight
        with torch.no_grad():
            import torch.nn.functional as F

            after_merge = F.linear(x, lora.weight, lora.bias)

        assert torch.allclose(before_merge, after_merge, atol=1e-5), (
            "merge 后输出应与 merge 前一致"
        )


class TestApplyLoRA:
    """测试 apply_lora 函数"""

    def test_apply_freezes_original(self, tiny_model):
        """apply_lora 应冻结原始参数"""
        apply_lora(tiny_model, rank=4, target_modules=["w_q", "w_v"])

        # 所有非 LoRA 参数应冻结
        for name, param in tiny_model.named_parameters():
            if "lora" not in name:
                assert not param.requires_grad, f"{name} 应被冻结"

    def test_only_lora_trainable(self, tiny_model):
        """只有 LoRA 参数应可训练"""
        apply_lora(tiny_model, rank=4, target_modules=["w_q", "w_v"])

        trainable = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
        assert len(trainable) > 0
        assert all("lora" in n for n in trainable)

    def test_lora_state_dict_keys(self, tiny_model):
        """lora_state_dict 应只包含 LoRA key"""
        apply_lora(tiny_model, rank=4, target_modules=["w_q", "w_v"])
        sd = lora_state_dict(tiny_model)

        assert len(sd) > 0
        for key in sd:
            assert "lora_A" in key or "lora_B" in key

    def test_lora_state_dict_roundtrip(self, tiny_config):
        """LoRA 参数保存/加载应保持一致"""
        model1 = GPT(tiny_config)
        apply_lora(model1, rank=4, target_modules=["w_q", "w_v"])

        # 手动修改 LoRA 参数
        for name, param in model1.named_parameters():
            if "lora" in name:
                param.data.fill_(0.42)

        sd = lora_state_dict(model1)

        model2 = GPT(tiny_config)
        apply_lora(model2, rank=4, target_modules=["w_q", "w_v"])
        load_lora_state_dict(model2, sd)

        sd2 = lora_state_dict(model2)
        for key in sd:
            assert torch.allclose(sd[key], sd2[key]), f"{key} 不一致"


class TestMergeLoRA:
    """测试 merge_lora 函数"""

    def test_merge_restores_structure(self, tiny_model, tiny_config):
        """merge 后 LoRALinear 应仍存在但权重已合并"""
        apply_lora(tiny_model, rank=4, target_modules=["w_q", "w_v"])
        merge_lora(tiny_model)

        # 模型应仍可正常推理
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _, _ = tiny_model(x)
        assert logits.shape[-1] == tiny_config.vocab_size

"""GPT 模型单元测试"""

import torch
import pytest

from model import ModelConfig, GPT


class TestGPT:
    """测试 GPT 模型"""

    def test_forward_shape(self, tiny_model, tiny_config, sample_input_ids):
        """前向传播输出维度应正确"""
        with torch.no_grad():
            logits, loss, _ = tiny_model(sample_input_ids)

        batch, seq_len = sample_input_ids.shape
        assert logits.shape == (batch, seq_len, tiny_config.vocab_size)
        assert loss is None  # 未提供 targets 时 loss 为 None

    def test_forward_with_targets(self, tiny_model, tiny_config, sample_input_ids):
        """提供 targets 时应计算 loss"""
        targets = sample_input_ids.clone()
        with torch.no_grad():
            logits, loss, _ = tiny_model(sample_input_ids, targets)

        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # loss 应为正数

    def test_gradient_flow(self, tiny_config, sample_input_ids):
        """梯度应正确回传"""
        model = GPT(tiny_config)
        model.train()

        targets = sample_input_ids.clone()
        logits, loss, _ = model(sample_input_ids, targets)
        loss.backward()

        # 检查至少有一个参数有梯度
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "反向传播应该产生梯度"

    def test_count_parameters(self, tiny_model, tiny_config):
        """模型实际参数量应与 config 估算一致 (数量级相同)"""
        actual = tiny_model.count_parameters()["total"]
        estimated = tiny_config.count_params()["total"]
        ratio = actual / estimated
        # 允许 50% 的差异 (weight tying 等)
        assert 0.5 < ratio < 2.0, (
            f"参数量差异过大: actual={actual}, estimated={estimated}"
        )

    def test_generate(self, tiny_model, tiny_config):
        """generate 应产生正确长度的序列"""
        from inference.generate import generate

        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        max_new = 8

        with torch.no_grad():
            output = generate(
                tiny_model, prompt, max_new_tokens=max_new, temperature=1.0, eos_token_id=-1
            )

        # 输出长度应至少包含 prompt 长度
        assert output.shape[0] == 1
        assert output.shape[1] >= 4
        assert output.shape[1] <= 4 + max_new

    def test_kv_cache_equivalence(self, tiny_model, tiny_config):
        """KV Cache 和无 Cache 的结果应一致"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            # 无 cache
            logits_no_cache, _, _ = tiny_model(prompt)
            last_no_cache = logits_no_cache[:, -1, :]

            # 有 cache: prefill 全部 token
            logits_cache, _, kv_caches = tiny_model(prompt, use_cache=True)
            last_cache = logits_cache[:, -1, :]

        # 最后一个 token 的 logits 应近似相等
        assert torch.allclose(last_no_cache, last_cache, atol=1e-4), (
            "KV Cache 的结果应与无 Cache 一致"
        )

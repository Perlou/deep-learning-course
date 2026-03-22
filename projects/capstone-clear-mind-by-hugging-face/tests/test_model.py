"""ClearMindForCausalLM 模型单元测试"""

import tempfile

import torch
import pytest
from transformers.modeling_outputs import CausalLMOutputWithPast

from model import ClearMindConfig, ClearMindForCausalLM


class TestClearMindForCausalLM:
    """测试 ClearMindForCausalLM"""

    def test_forward_shape(self, tiny_model, tiny_config, sample_input_ids):
        """前向传播 logits 形状应为 [batch, seq, vocab_size]"""
        with torch.no_grad():
            outputs = tiny_model(sample_input_ids)

        batch, seq_len = sample_input_ids.shape
        assert outputs.logits.shape == (batch, seq_len, tiny_config.vocab_size)
        assert outputs.loss is None  # 未提供 labels

    def test_forward_with_loss(self, tiny_model, sample_input_ids):
        """提供 labels 时应计算 loss"""
        labels = sample_input_ids.clone()
        with torch.no_grad():
            outputs = tiny_model(sample_input_ids, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.dim() == 0  # scalar
        assert outputs.loss.item() > 0

    def test_output_type(self, tiny_model, sample_input_ids):
        """返回类型应为 CausalLMOutputWithPast"""
        with torch.no_grad():
            outputs = tiny_model(sample_input_ids)
        assert isinstance(outputs, CausalLMOutputWithPast)

    def test_gradient_flow(self, tiny_config, sample_input_ids):
        """梯度应正确回传"""
        model = ClearMindForCausalLM(tiny_config)
        model.train()

        labels = sample_input_ids.clone()
        outputs = model(sample_input_ids, labels=labels)
        outputs.loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "反向传播应该产生梯度"

    def test_count_parameters(self, tiny_model, tiny_config):
        """实际参数量应与 config 估算一致 (数量级)"""
        actual = tiny_model.count_parameters()["total"]
        estimated = tiny_config.count_params()["total"]
        ratio = actual / estimated
        assert 0.5 < ratio < 2.0, (
            f"参数量差异过大: actual={actual}, estimated={estimated}"
        )

    def test_kv_cache_equivalence(self, tiny_model, tiny_config):
        """KV Cache 和无 Cache 的结果应一致"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            # 无 cache
            out_no_cache = tiny_model(prompt, use_cache=False)
            last_no_cache = out_no_cache.logits[:, -1, :]

            # 有 cache
            out_cache = tiny_model(prompt, use_cache=True)
            last_cache = out_cache.logits[:, -1, :]

        assert torch.allclose(last_no_cache, last_cache, atol=1e-4), (
            "KV Cache 的结果应与无 Cache 一致"
        )

    def test_generate(self, tiny_model, tiny_config):
        """model.generate() 应产生正确长度的序列"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        max_new = 8

        with torch.no_grad():
            output = tiny_model.generate(
                prompt,
                max_new_tokens=max_new,
                do_sample=False,
            )

        assert output.shape[0] == 1
        assert output.shape[1] >= 4
        assert output.shape[1] <= 4 + max_new

    def test_save_load_pretrained(self, tiny_model, tiny_config):
        """save_pretrained / from_pretrained 往返应一致"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 8))

        with torch.no_grad():
            original_logits = tiny_model(prompt).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)
            loaded = ClearMindForCausalLM.from_pretrained(tmpdir)
            loaded.eval()

        with torch.no_grad():
            loaded_logits = loaded(prompt).logits

        assert torch.allclose(original_logits, loaded_logits, atol=1e-5), (
            "加载后的模型输出应与原始一致"
        )

    def test_weight_tying(self, tiny_model):
        """embed_tokens 和 lm_head 权重应共享"""
        embed_weight = tiny_model.model.embed_tokens.weight
        lm_head_weight = tiny_model.lm_head.weight
        assert embed_weight.data_ptr() == lm_head_weight.data_ptr(), (
            "Weight tying: embed_tokens 和 lm_head 应共享同一块内存"
        )

    def test_kv_cache_incremental(self, tiny_model, tiny_config):
        """增量 decode: prefill + 逐步 decode 应与一次性计算一致"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 6))

        with torch.no_grad():
            # 一次性计算全部
            full_out = tiny_model(prompt, use_cache=False)
            full_logits = full_out.logits[:, -1, :]

            # Prefill 前 5 个 token
            prefill = tiny_model(prompt[:, :5], use_cache=True)
            kv_caches = prefill.past_key_values

            # Decode 第 6 个 token
            decode = tiny_model(prompt[:, 5:], past_key_values=kv_caches, use_cache=True)
            decode_logits = decode.logits[:, -1, :]

        assert torch.allclose(full_logits, decode_logits, atol=1e-4), (
            "增量 decode 应与一次性计算一致"
        )

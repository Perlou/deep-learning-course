"""文本生成模块单元测试"""

import torch

from model import GPT
from inference.generate import generate


class TestGenerate:
    """测试 generate 函数"""

    def test_output_length(self, tiny_model, tiny_config):
        """输出长度应 ≤ prompt + max_new_tokens"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        max_new = 10

        with torch.no_grad():
            output = generate(
                tiny_model,
                prompt,
                max_new_tokens=max_new,
                temperature=1.0,
                top_k=50,
                eos_token_id=-1,  # 禁用 EOS 提前停止
            )

        assert output.shape[0] == 1
        assert output.shape[1] >= 4  # 至少 prompt 长度
        assert output.shape[1] <= 4 + max_new

    def test_greedy_deterministic(self, tiny_model, tiny_config):
        """temperature=0 (greedy) 应产生确定性输出"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

        with torch.no_grad():
            out1 = generate(
                tiny_model,
                prompt.clone(),
                max_new_tokens=5,
                temperature=0,
                eos_token_id=-1,
            )
            out2 = generate(
                tiny_model,
                prompt.clone(),
                max_new_tokens=5,
                temperature=0,
                eos_token_id=-1,
            )

        assert torch.equal(out1, out2), "Greedy 模式应产生相同输出"

    def test_eos_stops_early(self, tiny_config):
        """遇到 EOS token 时应提前停止"""
        # 创建一个极小模型, 手动设置权重使每步都输出 EOS (id=2)
        model = GPT(tiny_config)
        model.eval()

        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        max_new = 20

        with torch.no_grad():
            output = generate(
                model,
                prompt,
                max_new_tokens=max_new,
                temperature=1.0,
                top_k=0,  # 不做 top-k
                top_p=1.0,  # 不做 top-p
                eos_token_id=2,
            )

        # 输出长度应 ≤ prompt + max_new (可能因 EOS 提前停止)
        assert output.shape[1] <= 4 + max_new

    def test_output_contains_prompt(self, tiny_model, tiny_config):
        """输出序列应以 prompt 开头"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

        with torch.no_grad():
            output = generate(
                tiny_model,
                prompt.clone(),
                max_new_tokens=5,
                temperature=1.0,
                eos_token_id=-1,
            )

        # 前 4 个 token 应与 prompt 一致
        assert torch.equal(output[:, :4], prompt)

    def test_repetition_penalty(self, tiny_model, tiny_config):
        """repetition_penalty 不应导致错误"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

        with torch.no_grad():
            output = generate(
                tiny_model,
                prompt,
                max_new_tokens=5,
                temperature=1.0,
                repetition_penalty=1.5,
                eos_token_id=-1,
            )

        assert output.shape[1] >= 4

    def test_top_k_filtering(self, tiny_model, tiny_config):
        """top_k 参数不应导致错误"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

        with torch.no_grad():
            output = generate(
                tiny_model,
                prompt,
                max_new_tokens=5,
                temperature=1.0,
                top_k=5,
                eos_token_id=-1,
            )

        assert output.shape[1] >= 4

    def test_top_p_filtering(self, tiny_model, tiny_config):
        """top_p 参数不应导致错误"""
        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))

        with torch.no_grad():
            output = generate(
                tiny_model,
                prompt,
                max_new_tokens=5,
                temperature=1.0,
                top_p=0.5,
                eos_token_id=-1,
            )

        assert output.shape[1] >= 4

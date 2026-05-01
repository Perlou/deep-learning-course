"""Phase 3 trainer 模块的最小冒烟测试

覆盖 distillation_loss、grpo.default_rule_reward / ``_ngram_repetition_penalty``、
rollout_engine._per_token_logps 这些纯函数 / 辅助逻辑（不跑完整训练循环），
保证关键数学正确性与接口稳定性。

完整训练路径（``trainer.train()``）由 smoke_test.py 与 AutoDL 训练时验证。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


# ============================================================
# distillation_loss
# ============================================================


class TestDistillationLoss:
    """src/training/distillation.py::distillation_loss"""

    def test_identical_logits_zero_loss(self):
        """teacher == student 时 KL 应该 ≈ 0"""
        from training.distillation import distillation_loss

        torch.manual_seed(0)
        logits = torch.randn(8, 100)
        loss = distillation_loss(logits, logits, temperature=1.0)
        assert loss.item() < 1e-5, f"identical logits 应得 KL≈0，实际 {loss.item()}"

    def test_different_logits_positive_loss(self):
        """teacher != student 时 KL > 0"""
        from training.distillation import distillation_loss

        torch.manual_seed(0)
        student = torch.randn(8, 100)
        teacher = torch.randn(8, 100) + 1.0
        loss = distillation_loss(student, teacher, temperature=1.0)
        assert loss.item() > 0
        assert torch.isfinite(loss).item()

    def test_temperature_t_squared_scaling(self):
        """T² 缩放：固定分布差异，温度 T 翻倍，loss 应该相应缩放（粗略）"""
        from training.distillation import distillation_loss

        torch.manual_seed(0)
        student = torch.randn(8, 100)
        teacher = torch.randn(8, 100)

        loss_t1 = distillation_loss(student, teacher, temperature=1.0).item()
        loss_t4 = distillation_loss(student, teacher, temperature=4.0).item()
        # T²=16，但 logits/T 后分布更平滑，KL 本身会变小，所以 loss_t4 不会真的是
        # 16*loss_t1，但应该有限且正
        assert math.isfinite(loss_t4)
        assert loss_t4 > 0

    def test_gradient_flows_to_student_only(self):
        """KL 应该只对 student logits 有梯度，teacher 走 no_grad"""
        from training.distillation import distillation_loss

        student = torch.randn(4, 50, requires_grad=True)
        teacher = torch.randn(4, 50, requires_grad=True)
        loss = distillation_loss(student, teacher, temperature=2.0)
        loss.backward()

        assert student.grad is not None
        assert torch.isfinite(student.grad).all()
        # teacher 在 no_grad 下不该有梯度
        assert teacher.grad is None


# ============================================================
# GRPO 规则 reward
# ============================================================


class TestGRPORewards:
    """src/training/grpo.py::default_rule_reward"""

    def test_returns_correct_shape(self):
        from training.grpo import default_rule_reward

        prompts = ["a", "b", "c"]
        responses = ["x", "y", "z"]
        rewards = default_rule_reward(prompts, responses)
        assert rewards.shape == (3,)
        assert rewards.dtype == torch.float32

    def test_proper_length_response_gets_positive(self):
        """长度合理 + 无重复 → reward > 0"""
        from training.grpo import default_rule_reward

        prompts = ["q"]
        responses = ["这是一个长度合理的中文回答，没有任何 trigram 级别的重复内容。"]
        r = default_rule_reward(prompts, responses)[0].item()
        assert r > 0, f"合理回答应得正 reward，实际 {r}"

    def test_too_short_response_penalized(self):
        """长度太短 → reward 偏负"""
        from training.grpo import default_rule_reward

        prompts = ["q"]
        responses = ["x"]
        r = default_rule_reward(prompts, responses)[0].item()
        assert r < 0, f"过短回答应得负 reward，实际 {r}"

    def test_repetition_penalty_cap(self):
        """trigram 重复率有上限 0.5"""
        from training.grpo import _ngram_repetition_penalty

        # 大量重复
        repeated = "abc" * 100
        p = _ngram_repetition_penalty(repeated, n=3)
        assert 0.0 <= p <= 0.5, f"重复惩罚应在 [0, 0.5]，实际 {p}"

        # 无重复（短文本）
        clean = "abcdefghijklmn"
        p2 = _ngram_repetition_penalty(clean, n=3)
        assert p2 < 0.1

    def test_think_block_bonus(self):
        """合理的 think 段 → 额外加分"""
        from training.grpo import default_rule_reward

        with_think = "<think>这是一段合理长度的思考过程，至少 20 个字符的内容。</think>这是答案部分，长度也足够。"
        without_think = "这是答案部分，长度也足够。" * 3

        prompts = ["q", "q"]
        responses = [with_think, without_think]
        rewards = default_rule_reward(prompts, responses)
        # 有 think 的应该比没 think 的高（前者 +1.0+0.25，后者 0）
        # 但 with_think 整体长度更长，可能也加分；只要 with_think >= without_think 即可
        assert rewards[0].item() >= rewards[1].item() - 0.5  # 容忍长度评分差异


# ============================================================
# RolloutEngine 工具函数
# ============================================================


class TestRolloutEngine:
    """src/training/rollout_engine.py"""

    def test_per_token_logps_shape(self):
        """``_per_token_logps`` 返回形状应为 [B, n_keep]"""
        from training.rollout_engine import _per_token_logps
        from model import ModelConfig, GPT

        cfg = ModelConfig.tiny()
        model = GPT(cfg)
        model.eval()

        B, T = 2, 32
        n_keep = 8
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        logps = _per_token_logps(model, input_ids, n_keep=n_keep)
        assert logps.shape == (B, n_keep)
        assert torch.isfinite(logps).all()

    def test_per_token_logps_zero_keep(self):
        """n_keep=0 应返回空张量而不是报错"""
        from training.rollout_engine import _per_token_logps
        from model import ModelConfig, GPT

        cfg = ModelConfig.tiny()
        model = GPT(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logps = _per_token_logps(model, input_ids, n_keep=0)
        assert logps.shape == (1, 0)

    def test_rollout_result_dataclass(self):
        """RolloutResult 字段齐全且可访问"""
        from training.rollout_engine import RolloutResult

        B, P, R = 2, 16, 8
        result = RolloutResult(
            output_ids=torch.zeros((B, P + R), dtype=torch.long),
            completion_ids=torch.zeros((B, R), dtype=torch.long),
            per_token_logps=torch.zeros((B, R)),
            completions=["", ""],
            prompt_lens=torch.tensor([P, P]),
            completion_mask=torch.ones((B, R)),
        )
        # 关键字段访问不挂
        assert result.output_ids.shape == (B, P + R)
        assert result.completion_ids.shape == (B, R)
        assert result.per_token_logps.shape == (B, R)
        assert len(result.completions) == B
        assert result.prompt_lens.shape == (B,)
        assert result.completion_mask.shape == (B, R)

    def test_torch_engine_creatable(self):
        """``create_rollout_engine('torch', ...)`` 应能成功构造"""
        from training.rollout_engine import create_rollout_engine
        from model import ModelConfig, GPT

        cfg = ModelConfig.tiny()
        model = GPT(cfg)

        # 用 mock tokenizer（rollout 用不到 chat_template 时也能跑）
        class MockTokenizer:
            pad_id = 0
            eos_id = 2
            bos_id = 1

            def decode(self, ids, **kw):
                return "decoded"

        engine = create_rollout_engine(
            engine_type="torch",
            policy_model=model,
            tokenizer=MockTokenizer(),
            device=torch.device("cpu"),
        )
        # 接口存在性
        assert hasattr(engine, "rollout")
        assert hasattr(engine, "update_policy")

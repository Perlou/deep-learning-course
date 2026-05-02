"""RMSNorm 单元测试

重点验证 dtype-safety：bf16/fp16 输入下，pow(2).mean(-1) 必须在 fp32 内部计算，
否则长序列 / 大方差输入会丢精度，触发 loss 尖刺。

参考实现：minimind/model/model_minimind.py:60
    return (self.weight * self.norm(x.float())).type_as(x)
"""

import pytest
import torch

from model.normalization import RMSNorm


class TestRMSNormDtype:
    def test_output_dtype_matches_input_fp32(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64, dtype=torch.float32)
        out = norm(x)
        assert out.dtype == torch.float32
        assert out.shape == x.shape

    def test_output_dtype_matches_input_bf16(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64).to(torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_bf16_internal_compute_in_fp32(self):
        """bf16 输入下 RMSNorm 应在 fp32 中计算 RMS，再 cast 回 bf16。

        构造大方差输入放大 bf16 累积误差：旧实现（全 bf16 计算）的 rms 与
        fp32 参考偏差较大；新实现应与 fp32 参考完全一致。
        """
        torch.manual_seed(0)
        dim = 512
        norm = RMSNorm(dim=dim, eps=1e-6)
        # 大方差 + 大 dim：pow(2).mean 在 bf16 下累积舍入误差
        x_fp32 = torch.randn(8, dim) * 5.0
        x_bf16 = x_fp32.to(torch.bfloat16)

        out = norm(x_bf16)

        # fp32 参考：内部全 fp32 计算，最后 cast
        x_in_fp32 = x_bf16.float()
        rms_ref = (
            x_in_fp32.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        )
        ref = (x_in_fp32 * rms_ref * norm.weight.float()).to(torch.bfloat16)

        assert out.dtype == torch.bfloat16
        # 允许极小数值误差但不允许 dtype 误差累积
        assert torch.equal(out, ref), (
            "RMSNorm 在 bf16 下输出应等价于 fp32 内部计算后 cast 回 bf16；"
            f"max diff={ (out.float() - ref.float()).abs().max().item():.6f}"
        )

    def test_fp16_internal_compute_in_fp32(self):
        """fp16 同样需要内部 fp32：pow(2) 在 fp16 下范围更窄（最大 ~65504）"""
        torch.manual_seed(1)
        dim = 256
        norm = RMSNorm(dim=dim, eps=1e-6)
        x_fp32 = torch.randn(4, dim) * 2.0
        x_fp16 = x_fp32.to(torch.float16)

        out = norm(x_fp16)

        x_in_fp32 = x_fp16.float()
        rms_ref = (
            x_in_fp32.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        )
        ref = (x_in_fp32 * rms_ref * norm.weight.float()).to(torch.float16)

        assert out.dtype == torch.float16
        assert torch.equal(out, ref)

    def test_numerical_correctness_fp32(self):
        """fp32 下应满足 RMS(out / weight) ≈ 1（基础数值正确性）"""
        torch.manual_seed(2)
        norm = RMSNorm(dim=128, eps=1e-6)
        x = torch.randn(4, 128) * 3.0
        out = norm(x)
        # 抵消 weight=1 的初始化影响
        rms_out = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-4)

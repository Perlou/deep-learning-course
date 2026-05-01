"""SkipBatchSampler 单元测试"""

import pytest

from training.trainer_utils import SkipBatchSampler


class TestSkipBatchSampler:
    def test_no_skip(self):
        """skip_batches=0 时与普通 batch sampler 等价"""
        idx = list(range(10))
        sampler = SkipBatchSampler(idx, batch_size=2, skip_batches=0)
        batches = list(sampler)
        assert batches == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def test_skip_2_batches(self):
        """skip_batches=2 应跳过前 4 个样本"""
        idx = list(range(10))
        sampler = SkipBatchSampler(idx, batch_size=2, skip_batches=2)
        batches = list(sampler)
        assert batches == [[4, 5], [6, 7], [8, 9]]

    def test_skip_more_than_data(self):
        """skip_batches > 数据总量 时返回空"""
        idx = list(range(10))
        sampler = SkipBatchSampler(idx, batch_size=2, skip_batches=99)
        batches = list(sampler)
        assert batches == []

    def test_partial_last_batch(self):
        """末尾不足一个 batch 时也应返回（不丢失数据）"""
        idx = list(range(7))
        sampler = SkipBatchSampler(idx, batch_size=3, skip_batches=1)
        batches = list(sampler)
        # skip 第 1 个 batch=[0,1,2]，剩 [3,4,5] + [6]
        assert batches == [[3, 4, 5], [6]]

    def test_len(self):
        """__len__ 返回剩余 batch 数"""
        idx = list(range(10))
        sampler = SkipBatchSampler(idx, batch_size=2, skip_batches=2)
        assert len(sampler) == 3

    def test_negative_skip_treated_as_zero(self):
        """负数 skip 应当作 0 处理"""
        idx = list(range(6))
        sampler = SkipBatchSampler(idx, batch_size=2, skip_batches=-5)
        batches = list(sampler)
        assert batches == [[0, 1], [2, 3], [4, 5]]

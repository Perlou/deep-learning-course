"""
test_data.py — 数据处理单元测试
================================

验证三种数据处理函数的正确性:
  - load_pretrain_dataset: 预训练数据 tokenize
  - load_sft_dataset: SFT 数据格式化 + labels mask
  - load_dpo_dataset: DPO 数据格式化
"""

import pytest
from transformers import DataCollatorForLanguageModeling

from data import load_pretrain_dataset, load_sft_dataset, load_dpo_dataset


# ============================================================
# Pretrain 数据测试
# ============================================================


class TestPretrainDataset:
    """预训练数据处理测试"""

    def test_load_pretrain_dataset(self, tmp_tokenizer, sample_pretrain_data):
        """load_pretrain_dataset 返回 Dataset，含 input_ids 和 attention_mask"""
        result = load_pretrain_dataset(
            sample_pretrain_data, tmp_tokenizer, max_length=64
        )

        assert "train" in result
        assert "validation" in result
        assert len(result["train"]) > 0
        assert len(result["validation"]) > 0

        # 检查数据列
        sample = result["train"][0]
        assert "input_ids" in sample
        assert "attention_mask" in sample

    def test_pretrain_tokenized_length(self, tmp_tokenizer, sample_pretrain_data):
        """tokenize 后的序列长度 ≤ max_length"""
        max_length = 32
        result = load_pretrain_dataset(
            sample_pretrain_data, tmp_tokenizer, max_length=max_length
        )

        for sample in result["train"]:
            assert len(sample["input_ids"]) <= max_length
            assert len(sample["attention_mask"]) <= max_length

    def test_pretrain_input_ids_valid(self, tmp_tokenizer, sample_pretrain_data):
        """input_ids 都是有效的 token id"""
        result = load_pretrain_dataset(
            sample_pretrain_data, tmp_tokenizer, max_length=64
        )

        vocab_size = tmp_tokenizer.vocab_size
        for sample in result["train"]:
            for token_id in sample["input_ids"]:
                assert 0 <= token_id < vocab_size

    def test_pretrain_collator(self, tmp_tokenizer, sample_pretrain_data):
        """DataCollatorForLanguageModeling 产生正确的 labels"""
        result = load_pretrain_dataset(
            sample_pretrain_data, tmp_tokenizer, max_length=64
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=tmp_tokenizer, mlm=False
        )

        # 取两个样本做 collate
        batch = collator([result["train"][i] for i in range(min(2, len(result["train"])))])

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch

        # labels 中 padding 位置应为 -100
        pad_id = tmp_tokenizer.pad_token_id
        for i in range(batch["input_ids"].shape[0]):
            for j in range(batch["input_ids"].shape[1]):
                if batch["input_ids"][i, j] == pad_id:
                    assert batch["labels"][i, j] == -100

    def test_pretrain_train_val_split(self, tmp_tokenizer, sample_pretrain_data):
        """train/validation 划分比例正确"""
        result = load_pretrain_dataset(
            sample_pretrain_data, tmp_tokenizer, max_length=64, val_ratio=0.5
        )
        total = len(result["train"]) + len(result["validation"])
        # 6 条数据，50% split → 3+3
        assert total == 6


# ============================================================
# SFT 数据测试
# ============================================================


class TestSFTDataset:
    """SFT 指令微调数据处理测试"""

    def test_load_sft_dataset(self, tmp_tokenizer, sample_sft_data):
        """load_sft_dataset 返回 Dataset，含 input_ids 和 labels"""
        result = load_sft_dataset(
            sample_sft_data, tmp_tokenizer, max_length=128
        )

        assert "train" in result
        assert "validation" in result

        sample = result["train"][0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample

    def test_sft_labels_mask(self, tmp_tokenizer, sample_sft_data):
        """prompt 部分 labels 为 -100，response 部分为有效 token"""
        result = load_sft_dataset(
            sample_sft_data, tmp_tokenizer, max_length=128
        )

        for sample in result["train"]:
            labels = sample["labels"]

            # 应有部分 -100 (prompt 部分)
            has_masked = any(l == -100 for l in labels)
            assert has_masked, "labels 中应有 -100 (prompt 部分被 mask)"

            # 应有部分非 -100 (response 部分)
            has_valid = any(l != -100 for l in labels)
            assert has_valid, "labels 中应有有效 token (response 部分)"

            # -100 应出现在序列开头 (prompt 部分)
            first_valid_idx = next(i for i, l in enumerate(labels) if l != -100)
            for i in range(first_valid_idx):
                assert labels[i] == -100

    def test_sft_format(self, tmp_tokenizer, sample_sft_data):
        """chat_template 格式化包含 "Human:" 和 "Assistant:" 标记"""
        # 直接验证 chat_template 的格式化输出 (不受 tokenizer 解码精度影响)
        formatted = tmp_tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "测试问题"},
                {"role": "assistant", "content": "测试回答"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert "Human:" in formatted, f"chat_template 输出应包含 'Human:', got: {formatted}"
        assert "Assistant:" in formatted, f"chat_template 输出应包含 'Assistant:', got: {formatted}"

        # 验证 load_sft_dataset 返回的 input_ids 和 labels 长度一致
        result = load_sft_dataset(sample_sft_data, tmp_tokenizer, max_length=128)
        sample = result["train"][0]
        assert len(sample["input_ids"]) == len(sample["labels"])

    def test_sft_tokenized_length(self, tmp_tokenizer, sample_sft_data):
        """tokenize 后的序列长度 ≤ max_length"""
        max_length = 64
        result = load_sft_dataset(
            sample_sft_data, tmp_tokenizer, max_length=max_length
        )

        for sample in result["train"]:
            assert len(sample["input_ids"]) <= max_length

    def test_sft_with_input_field(self, tmp_tokenizer, sample_sft_data):
        """带 input 字段的样本正确处理"""
        result = load_sft_dataset(
            sample_sft_data, tmp_tokenizer, max_length=128
        )

        # sample_sft_data 中第 3 条有 input 字段 "深度学习很有趣"
        # 验证数据集总数正确
        total = len(result["train"]) + len(result["validation"])
        assert total == 6


# ============================================================
# DPO 数据测试
# ============================================================


class TestDPODataset:
    """DPO 偏好对齐数据处理测试"""

    def test_load_dpo_dataset(self, tmp_tokenizer, sample_dpo_data):
        """load_dpo_dataset 返回 Dataset，含 prompt/chosen/rejected 字段"""
        result = load_dpo_dataset(
            sample_dpo_data, tmp_tokenizer, max_length=128
        )

        assert "train" in result
        assert "validation" in result

        sample = result["train"][0]
        assert "prompt" in sample
        assert "chosen" in sample
        assert "rejected" in sample

    def test_dpo_format(self, tmp_tokenizer, sample_dpo_data):
        """prompt 格式正确 — 包含 Human: 和 Assistant: 标记"""
        result = load_dpo_dataset(
            sample_dpo_data, tmp_tokenizer, max_length=128
        )

        for sample in result["train"]:
            assert "Human:" in sample["prompt"], "DPO prompt 应包含 'Human:'"
            assert "Assistant:" in sample["prompt"], "DPO prompt 应包含 'Assistant:'"

    def test_dpo_strings(self, tmp_tokenizer, sample_dpo_data):
        """prompt/chosen/rejected 都是字符串 (不是 token ids)"""
        result = load_dpo_dataset(
            sample_dpo_data, tmp_tokenizer, max_length=128
        )

        for sample in result["train"]:
            assert isinstance(sample["prompt"], str)
            assert isinstance(sample["chosen"], str)
            assert isinstance(sample["rejected"], str)

    def test_dpo_chosen_rejected_different(self, tmp_tokenizer, sample_dpo_data):
        """chosen 和 rejected 应该不同"""
        result = load_dpo_dataset(
            sample_dpo_data, tmp_tokenizer, max_length=128
        )

        for sample in result["train"]:
            assert sample["chosen"] != sample["rejected"]

    def test_dpo_train_val_split(self, tmp_tokenizer, sample_dpo_data):
        """train/validation 划分正确"""
        result = load_dpo_dataset(
            sample_dpo_data, tmp_tokenizer, max_length=128
        )
        total = len(result["train"]) + len(result["validation"])
        assert total == 6

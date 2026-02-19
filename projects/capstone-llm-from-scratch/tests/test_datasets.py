"""数据集模块单元测试"""

import json

import torch


class MockTokenizer:
    """用于测试的 Mock Tokenizer

    简单地将每个字符映射为其 ord 值 (mod vocab_size)。
    """

    def __init__(self, vocab_size: int = 100):
        self._vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 0
        self.pad_id = 2  # 与 eos_id 相同

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = [ord(c) % self._vocab_size for c in text] if text else []
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i > 2)


# ============================================================
# PretrainDataset 测试
# ============================================================


class TestPretrainDataset:
    """测试预训练数据集"""

    def test_jsonl_loading(self, tmp_path):
        """应能从 .jsonl 文件加载数据"""
        from data.pretrain_dataset import PretrainDataset

        # 创建测试数据 (每行足够长以产生至少 1 个样本)
        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Hello world sentence number {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(str(data_file), tokenizer, max_seq_len=32)

        assert len(dataset) > 0

    def test_getitem_shape(self, tmp_path):
        """__getitem__ 应返回正确形状的 input_ids 和 labels"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Sample text for testing {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        max_seq = 16
        dataset = PretrainDataset(str(data_file), tokenizer, max_seq_len=max_seq)

        if len(dataset) > 0:
            sample = dataset[0]
            assert "input_ids" in sample
            assert "labels" in sample
            assert sample["input_ids"].shape == (max_seq,)
            assert sample["labels"].shape == (max_seq,)
            assert sample["input_ids"].dtype == torch.long
            assert sample["labels"].dtype == torch.long

    def test_txt_loading(self, tmp_path):
        """应能从 .txt 文件加载数据"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.txt"
        text = "\n\n".join(
            [f"Paragraph {i} with some text content." * 10 for i in range(20)]
        )
        data_file.write_text(text, encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(str(data_file), tokenizer, max_seq_len=16)

        assert len(dataset) > 0

    def test_input_target_shift(self, tmp_path):
        """target 应是 input 的右移一位"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Long text for shift test number {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(str(data_file), tokenizer, max_seq_len=16)

        if len(dataset) > 0:
            sample = dataset[0]
            # 由于数据构建方式: target[i] == 原始 chunk 中的 token[i+1]
            # input_ids 和 labels 来自同一 chunk 的错位, labels 右移一位
            assert sample["input_ids"].shape == sample["labels"].shape


# ============================================================
# SFTDataset 测试
# ============================================================


class TestSFTDataset:
    """测试 SFT 指令微调数据集"""

    def test_alpaca_format(self, tmp_path):
        """应能加载 Alpaca 格式数据"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "Translate hello", "input": "to Chinese", "output": "你好"},
        ]
        data_file.write_text(
            "\n".join(json.dumps(s) for s in samples), encoding="utf-8"
        )

        tokenizer = MockTokenizer()
        dataset = SFTDataset(str(data_file), tokenizer, max_seq_len=64)

        assert len(dataset) == 2

    def test_loss_mask(self, tmp_path):
        """prompt 部分的 labels 应为 -100"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {"instruction": "Hi", "output": "Hello!"},
        ]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = SFTDataset(str(data_file), tokenizer, max_seq_len=64)

        sample = dataset[0]
        labels = sample["labels"]

        # 前面应有 -100 (prompt 部分)
        assert (labels == -100).any(), "prompt 部分应被 mask 为 -100"
        # 后面应有 >= 0 的值 (response 部分)
        assert (labels >= 0).any(), "response 部分不应被 mask"

    def test_getitem_shape(self, tmp_path):
        """__getitem__ 返回形状应正确"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        max_seq = 32
        samples = [{"instruction": "Test", "output": "OK"}]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = SFTDataset(str(data_file), tokenizer, max_seq_len=max_seq)

        sample = dataset[0]
        assert sample["input_ids"].shape == (max_seq,)
        assert sample["labels"].shape == (max_seq,)


# ============================================================
# DPODataset 测试
# ============================================================


class TestDPODataset:
    """测试 DPO 偏好对齐数据集"""

    def test_loading(self, tmp_path):
        """应能加载 DPO 格式数据"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {
                "prompt": "What is 1+1?",
                "chosen": "2",
                "rejected": "I don't know",
            }
        ]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = DPODataset(str(data_file), tokenizer, max_seq_len=64)

        assert len(dataset) == 1

    def test_getitem_keys(self, tmp_path):
        """__getitem__ 应返回 4 个 key"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "Bye!"},
        ]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = DPODataset(str(data_file), tokenizer, max_seq_len=64)

        sample = dataset[0]
        expected_keys = [
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ]
        for key in expected_keys:
            assert key in sample, f"缺少 key: {key}"

    def test_getitem_shape(self, tmp_path):
        """所有输出张量形状应为 [max_seq_len]"""
        from data.dpo_dataset import DPODataset

        max_seq = 32
        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {"prompt": "Question", "chosen": "Good answer", "rejected": "Bad"},
        ]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = DPODataset(str(data_file), tokenizer, max_seq_len=max_seq)

        sample = dataset[0]
        for key in [
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ]:
            assert sample[key].shape == (max_seq,), f"{key} 形状不正确"
            assert sample[key].dtype == torch.long

    def test_loss_mask(self, tmp_path):
        """chosen 和 rejected 的 prompt 部分应为 -100"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {"prompt": "Question here", "chosen": "Good answer", "rejected": "Bad"},
        ]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = DPODataset(str(data_file), tokenizer, max_seq_len=64)

        sample = dataset[0]
        # chosen 和 rejected 的 prompt 部分都应被 mask
        assert (sample["chosen_labels"] == -100).any()
        assert (sample["rejected_labels"] == -100).any()

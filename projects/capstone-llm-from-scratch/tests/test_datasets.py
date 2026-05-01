"""数据集模块单元测试

PretrainDataset 走轻量 MockTokenizer（无需 HF tokenizer 文件）；
SFT / DPO 基于 minimind tokenizer + chat_template，使用 conftest 提供的 ``hf_tokenizer``。
"""

import json

import pytest
import torch


class MockTokenizer:
    """用于 PretrainDataset 测试的轻量 tokenizer

    将每个字符映射为其 ord 值 (mod vocab_size)。不实现 chat_template。
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
# PretrainDataset 测试（packed 模式 + per_sample 模式）
# ============================================================


class TestPretrainDataset:
    """测试预训练数据集"""

    def test_jsonl_loading_packed(self, tmp_path):
        """packed 模式应能从 .jsonl 文件加载数据"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Hello world sentence number {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(
            str(data_file), tokenizer, max_seq_len=32, mode="packed"
        )
        assert len(dataset) > 0

    def test_getitem_shape_packed(self, tmp_path):
        """packed 模式 __getitem__ 应返回 [max_seq_len] 的 input_ids 和 labels"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Sample text for testing {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        max_seq = 16
        dataset = PretrainDataset(
            str(data_file), tokenizer, max_seq_len=max_seq, mode="packed"
        )
        sample = dataset[0]
        assert "input_ids" in sample and "labels" in sample
        assert sample["input_ids"].shape == (max_seq,)
        assert sample["labels"].shape == (max_seq,)
        assert sample["input_ids"].dtype == torch.long

    def test_txt_loading_packed(self, tmp_path):
        """packed 模式应能从 .txt 文件加载数据"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.txt"
        text = "\n\n".join(
            [f"Paragraph {i} with some text content." * 10 for i in range(20)]
        )
        data_file.write_text(text, encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(
            str(data_file), tokenizer, max_seq_len=16, mode="packed"
        )
        assert len(dataset) > 0

    def test_per_sample_mode_shape(self, tmp_path):
        """per_sample 模式 __getitem__ 应返回 [max_seq_len] 张量，pad 处 labels=-100"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        # 第一条很短（会触发 padding），第二条很长（会触发截断）
        samples = [{"text": "hi"}, {"text": "x" * 200}]
        data_file.write_text(
            "\n".join(json.dumps(s) for s in samples), encoding="utf-8"
        )

        tokenizer = MockTokenizer()
        max_seq = 32
        dataset = PretrainDataset(
            str(data_file), tokenizer, max_seq_len=max_seq, mode="per_sample"
        )
        assert len(dataset) == 2
        s0 = dataset[0]
        assert s0["input_ids"].shape == (max_seq,)
        assert s0["labels"].shape == (max_seq,)
        # 第一条很短，必有 padding，因而 labels 中至少有一个 -100
        assert (s0["labels"] == -100).any()

    def test_input_target_shift_packed(self, tmp_path):
        """packed 模式 labels 应是 input_ids 的右移一位"""
        from data.pretrain_dataset import PretrainDataset

        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"text": f"Long text for shift test number {i} " * 10})
            for i in range(20)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        tokenizer = MockTokenizer()
        dataset = PretrainDataset(
            str(data_file), tokenizer, max_seq_len=16, mode="packed"
        )
        if len(dataset) > 0:
            sample = dataset[0]
            assert sample["input_ids"].shape == sample["labels"].shape


# ============================================================
# SFTDataset 测试（HF tokenizer + chat_template）
# ============================================================


class TestSFTDataset:
    """测试 SFT 指令微调数据集"""

    def test_conversations_loading(self, tmp_path, hf_tokenizer):
        """应能加载 minimind 风格 conversations 数据"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "1+1=?"},
                    {"role": "assistant", "content": "2"},
                ]
            },
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=128,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        assert len(dataset) == 2

    def test_alpaca_fallback(self, tmp_path, hf_tokenizer):
        """Alpaca 格式应自动转 conversations"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {"instruction": "What is AI?", "output": "Artificial Intelligence."},
            {
                "instruction": "Translate hello",
                "input": "to Chinese",
                "output": "你好",
            },
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=128,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        assert len(dataset) == 2

    def test_loss_mask(self, tmp_path, hf_tokenizer):
        """assistant 段应能被 mask 扫描出来，user/system 段保持 -100"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好世界"},
                ]
            }
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=128,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        sample = dataset[0]
        labels = sample["labels"]
        # 必须有 user 部分被 mask
        assert (labels == -100).any(), "user/system/padding 部分应被 mask 为 -100"
        # 必须有 assistant 部分（>=0 的 label）
        assert (labels >= 0).any(), "assistant 部分不应被 mask"

    def test_tool_calls_string_format(self, tmp_path, hf_tokenizer):
        """minimind 数据中 tool_calls 字段可能是 JSON 字符串，应自动 json.loads"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        # 模拟 minimind 风格：tool_calls 是字符串
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "查询北京天气"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": json.dumps(
                            [
                                {
                                    "name": "get_current_weather",
                                    "arguments": {"location": "北京"},
                                }
                            ]
                        ),
                    },
                    {"role": "tool", "content": "{\"temperature\": \"28°C\"}"},
                    {"role": "assistant", "content": "北京当前 28°C"},
                ]
            }
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=256,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        # 不应抛异常
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample

    def test_tool_calls_malformed_skipped(self, tmp_path, hf_tokenizer):
        """tool_calls 中缺 name/arguments 的项应被跳过，不应让整个样本失败"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "Q"},
                    {
                        "role": "assistant",
                        "content": "A",
                        # 缺 arguments 字段，应被过滤
                        "tool_calls": [{"name": "broken_tool"}],
                    },
                ]
            }
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=128,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        # 不应抛异常
        sample = dataset[0]
        assert "input_ids" in sample

    def test_getitem_shape(self, tmp_path, hf_tokenizer):
        """__getitem__ 返回形状应为 [max_seq_len]"""
        from data.sft_dataset import SFTDataset

        data_file = tmp_path / "sft.jsonl"
        samples = [
            {
                "conversations": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "OK"},
                ]
            }
        ]
        data_file.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples),
            encoding="utf-8",
        )

        max_seq = 64
        dataset = SFTDataset(
            str(data_file),
            hf_tokenizer,
            max_seq_len=max_seq,
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
        )
        sample = dataset[0]
        assert sample["input_ids"].shape == (max_seq,)
        assert sample["labels"].shape == (max_seq,)


# ============================================================
# DPODataset 测试（HF tokenizer + conversations）
# ============================================================


class TestDPODataset:
    """测试 DPO 偏好对齐数据集"""

    def test_conversations_format(self, tmp_path, hf_tokenizer):
        """应能加载 minimind 风格 chosen/rejected = conversations 列表"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "1+1=?"},
                    {"role": "assistant", "content": "2"},
                ],
                "rejected": [
                    {"role": "user", "content": "1+1=?"},
                    {"role": "assistant", "content": "I don't know"},
                ],
            }
        ]
        data_file.write_text(json.dumps(samples[0], ensure_ascii=False), encoding="utf-8")

        dataset = DPODataset(str(data_file), hf_tokenizer, max_seq_len=64)
        assert len(dataset) == 1

    def test_string_fallback(self, tmp_path, hf_tokenizer):
        """旧 prompt/chosen/rejected 字符串格式应被自动转换"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [{"prompt": "Hi", "chosen": "Hello!", "rejected": "Bye!"}]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        dataset = DPODataset(str(data_file), hf_tokenizer, max_seq_len=64)
        assert len(dataset) == 1

    def test_getitem_keys(self, tmp_path, hf_tokenizer):
        """__getitem__ 应返回 4 个 key"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [{"prompt": "Hi", "chosen": "Hello!", "rejected": "Bye!"}]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        dataset = DPODataset(str(data_file), hf_tokenizer, max_seq_len=64)
        sample = dataset[0]
        for key in (
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ):
            assert key in sample, f"缺少 key: {key}"

    def test_getitem_shape(self, tmp_path, hf_tokenizer):
        """所有输出张量形状应为 [max_seq_len]"""
        from data.dpo_dataset import DPODataset

        max_seq = 64
        data_file = tmp_path / "dpo.jsonl"
        samples = [{"prompt": "Question", "chosen": "Good answer", "rejected": "Bad"}]
        data_file.write_text(json.dumps(samples[0]), encoding="utf-8")

        dataset = DPODataset(str(data_file), hf_tokenizer, max_seq_len=max_seq)
        sample = dataset[0]
        for key in (
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ):
            assert sample[key].shape == (max_seq,), f"{key} 形状不正确"
            assert sample[key].dtype == torch.long

    def test_loss_mask(self, tmp_path, hf_tokenizer):
        """chosen 和 rejected 的非 assistant 部分应为 -100"""
        from data.dpo_dataset import DPODataset

        data_file = tmp_path / "dpo.jsonl"
        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "Question here"},
                    {"role": "assistant", "content": "Good answer"},
                ],
                "rejected": [
                    {"role": "user", "content": "Question here"},
                    {"role": "assistant", "content": "Bad"},
                ],
            }
        ]
        data_file.write_text(json.dumps(samples[0], ensure_ascii=False), encoding="utf-8")

        dataset = DPODataset(str(data_file), hf_tokenizer, max_seq_len=64)
        sample = dataset[0]
        # user 部分都该被 mask
        assert (sample["chosen_labels"] == -100).any()
        assert (sample["rejected_labels"] == -100).any()
        # assistant 部分必须保留
        assert (sample["chosen_labels"] >= 0).any()
        assert (sample["rejected_labels"] >= 0).any()

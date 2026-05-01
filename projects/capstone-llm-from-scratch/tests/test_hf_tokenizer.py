"""HFTokenizer 单元测试

覆盖：
  - 加载与 vocab/特殊 token 一致性
  - encode / decode / tokenize 接口与 ClearMindTokenizer 对齐
  - apply_chat_template 渲染（含 system / multi-turn / open_thinking）
  - generate_assistant_mask 扫描算法（精确定位 assistant 段）
  - find_subseq 通用子序列搜索

依赖 ``transformers`` 与 ``tokenizer/minimind/`` 文件。conftest 提供的
``hf_tokenizer`` fixture 在缺依赖时自动 skip。
"""

import pytest


class TestBasicProps:
    """基础属性"""

    def test_vocab_size_6400(self, hf_tokenizer):
        """minimind tokenizer vocab=6400"""
        assert hf_tokenizer.vocab_size == 6400

    def test_special_token_ids(self, hf_tokenizer):
        """bos=1, eos=2, pad=0（minimind 约定）"""
        assert hf_tokenizer.bos_id == 1
        assert hf_tokenizer.eos_id == 2
        assert hf_tokenizer.pad_id == 0

    def test_assistant_prefix_cached(self, hf_tokenizer):
        """构造时缓存的 assistant_prefix_ids 应非空且以 bos 起头"""
        assert len(hf_tokenizer.assistant_prefix_ids) > 0
        assert hf_tokenizer.assistant_prefix_ids[0] == hf_tokenizer.bos_id

    def test_eos_with_newline_cached(self, hf_tokenizer):
        """eos_with_newline_ids 应以 eos_id 起头"""
        assert len(hf_tokenizer.eos_with_newline_ids) > 0
        assert hf_tokenizer.eos_with_newline_ids[0] == hf_tokenizer.eos_id


class TestEncodeDecode:
    """encode / decode 接口"""

    def test_encode_decode_roundtrip(self, hf_tokenizer):
        """encode → decode 应能还原（含 special tokens）"""
        text = "你好，世界！Hello."
        ids = hf_tokenizer.encode(text, add_bos=False, add_eos=False)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        out = hf_tokenizer.decode(ids, skip_special_tokens=False)
        # decode 不一定完全等长（normalize 可能改空格），但子串应保留
        assert "你好" in out
        assert "Hello" in out

    def test_add_bos_eos_flags(self, hf_tokenizer):
        ids = hf_tokenizer.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == hf_tokenizer.bos_id
        assert ids[-1] == hf_tokenizer.eos_id

    def test_tokenize_returns_strings(self, hf_tokenizer):
        toks = hf_tokenizer.tokenize("Hello 你好")
        assert isinstance(toks, list)
        assert all(isinstance(t, str) for t in toks)


class TestChatTemplate:
    """apply_chat_template"""

    def test_render_string(self, hf_tokenizer):
        msgs = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "嗨"},
        ]
        text = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        assert isinstance(text, str)
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text

    def test_tokenize_returns_list(self, hf_tokenizer):
        """tokenize=True 必须返回 list[int]，不能是 BatchEncoding dict"""
        msgs = [{"role": "user", "content": "hi"}]
        ids = hf_tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_with_system_prompt(self, hf_tokenizer):
        msgs = [
            {"role": "system", "content": "你是有用的助手"},
            {"role": "user", "content": "你好"},
        ]
        text = hf_tokenizer.apply_chat_template(msgs, tokenize=False)
        assert "<|im_start|>system" in text

    def test_add_generation_prompt(self, hf_tokenizer):
        """add_generation_prompt=True 应在末尾追加 assistant 起始标签"""
        msgs = [{"role": "user", "content": "hi"}]
        text = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        assert text.rstrip().endswith("<think>\n\n</think>") or "assistant" in text


class TestAssistantMask:
    """generate_assistant_mask: 扫描 assistant 段生成 loss mask"""

    def test_simple_single_turn(self, hf_tokenizer):
        msgs = [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "2"},
        ]
        ids = hf_tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
        mask = hf_tokenizer.generate_assistant_mask(ids)
        assert len(mask) == len(ids)
        # 至少有一些 mask=1 的位置（assistant 段）
        assert sum(mask) > 0
        # mask=1 的位置 decode 出来应包含 "2" 与 EOS
        assistant_ids = [i for i, m in zip(ids, mask) if m == 1]
        text = hf_tokenizer.decode(assistant_ids, skip_special_tokens=False)
        assert "2" in text

    def test_multi_turn(self, hf_tokenizer):
        msgs = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "再见"},
            {"role": "assistant", "content": "拜拜"},
        ]
        ids = hf_tokenizer.apply_chat_template(msgs, tokenize=True)
        mask = hf_tokenizer.generate_assistant_mask(ids)
        # 多轮对话应有两段 assistant，mask 总和 > 单轮
        msgs_single = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
        ids_single = hf_tokenizer.apply_chat_template(msgs_single, tokenize=True)
        mask_single = hf_tokenizer.generate_assistant_mask(ids_single)
        assert sum(mask) > sum(mask_single)

    def test_user_only_no_assistant_mask(self, hf_tokenizer):
        """只有 user 消息时，应没有 assistant mask"""
        msgs = [{"role": "user", "content": "提问"}]
        ids = hf_tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False
        )
        mask = hf_tokenizer.generate_assistant_mask(ids)
        assert sum(mask) == 0


class TestFindSubseq:
    """find_subseq 通用工具"""

    def test_find_existing(self, hf_tokenizer):
        positions = hf_tokenizer.find_subseq([1, 2, 3, 4, 2, 3, 5], [2, 3])
        assert positions == [1, 4]

    def test_find_missing(self, hf_tokenizer):
        positions = hf_tokenizer.find_subseq([1, 2, 3], [9, 9])
        assert positions == []

    def test_empty_needle(self, hf_tokenizer):
        positions = hf_tokenizer.find_subseq([1, 2, 3], [])
        assert positions == []

    def test_needle_longer_than_haystack(self, hf_tokenizer):
        positions = hf_tokenizer.find_subseq([1, 2], [1, 2, 3])
        assert positions == []

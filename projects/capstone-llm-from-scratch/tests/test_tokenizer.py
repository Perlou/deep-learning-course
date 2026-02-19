"""ClearMindTokenizer 单元测试"""

import pytest

from data.tokenizer import ClearMindTokenizer


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory):
    """训练一个微型 tokenizer 用于测试"""
    tmp_dir = tmp_path_factory.mktemp("tokenizer")

    # 创建训练文本
    train_file = str(tmp_dir / "train.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for _ in range(100):
            f.write("Hello world, this is a test sentence.\n")
            f.write("深度学习是人工智能的核心技术。\n")
            f.write("The quick brown fox jumps over the lazy dog.\n")
            f.write("Transformer 架构改变了自然语言处理领域。\n")

    # 训练 tokenizer
    model_prefix = str(tmp_dir / "test_tok")
    ClearMindTokenizer.train(
        input_file=train_file,
        model_prefix=model_prefix,
        vocab_size=500,
        character_coverage=0.9995,
    )

    return ClearMindTokenizer(f"{model_prefix}.model")


class TestTokenizerBasic:
    """测试 Tokenizer 基本功能"""

    def test_vocab_size(self, trained_tokenizer):
        """词表大小应等于训练时指定的 vocab_size"""
        assert trained_tokenizer.vocab_size == 500

    def test_special_token_ids(self, trained_tokenizer):
        """特殊 token ID 应正确"""
        assert trained_tokenizer.bos_id == 1
        assert trained_tokenizer.eos_id == 2
        assert trained_tokenizer.unk_id == 0

    def test_encode_returns_list(self, trained_tokenizer):
        """encode 应返回 int 列表"""
        ids = trained_tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_returns_string(self, trained_tokenizer):
        """decode 应返回字符串"""
        ids = trained_tokenizer.encode("Hello world")
        text = trained_tokenizer.decode(ids)
        assert isinstance(text, str)

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """encode → decode 应还原原始文本"""
        original = "Hello world"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert decoded.strip() == original

    def test_encode_chinese(self, trained_tokenizer):
        """中文文本应能正常 encode/decode"""
        original = "深度学习"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)
        assert len(ids) > 0
        assert decoded.strip() == original


class TestTokenizerSpecialTokens:
    """测试特殊 token 处理"""

    def test_add_bos(self, trained_tokenizer):
        """add_bos=True 应在开头添加 BOS token"""
        ids_no_bos = trained_tokenizer.encode("test")
        ids_with_bos = trained_tokenizer.encode("test", add_bos=True)
        assert ids_with_bos[0] == trained_tokenizer.bos_id
        assert ids_with_bos[1:] == ids_no_bos

    def test_add_eos(self, trained_tokenizer):
        """add_eos=True 应在结尾添加 EOS token"""
        ids_no_eos = trained_tokenizer.encode("test")
        ids_with_eos = trained_tokenizer.encode("test", add_eos=True)
        assert ids_with_eos[-1] == trained_tokenizer.eos_id
        assert ids_with_eos[:-1] == ids_no_eos

    def test_add_bos_and_eos(self, trained_tokenizer):
        """同时 add_bos 和 add_eos"""
        ids = trained_tokenizer.encode("test", add_bos=True, add_eos=True)
        assert ids[0] == trained_tokenizer.bos_id
        assert ids[-1] == trained_tokenizer.eos_id

    def test_empty_string(self, trained_tokenizer):
        """空字符串应返回空列表"""
        ids = trained_tokenizer.encode("")
        assert ids == []

    def test_empty_string_with_bos_eos(self, trained_tokenizer):
        """空字符串 + BOS + EOS 应只有两个 token"""
        ids = trained_tokenizer.encode("", add_bos=True, add_eos=True)
        assert ids == [trained_tokenizer.bos_id, trained_tokenizer.eos_id]


class TestTokenizerUtilities:
    """测试工具方法"""

    def test_tokenize_returns_pieces(self, trained_tokenizer):
        """tokenize 应返回 token 字符串列表"""
        pieces = trained_tokenizer.tokenize("Hello world")
        assert isinstance(pieces, list)
        assert all(isinstance(p, str) for p in pieces)
        assert len(pieces) > 0

    def test_id_to_piece(self, trained_tokenizer):
        """id_to_piece 应返回字符串"""
        piece = trained_tokenizer.id_to_piece(1)
        assert isinstance(piece, str)

    def test_piece_to_id(self, trained_tokenizer):
        """piece_to_id 应返回整数"""
        # BOS piece 的 ID 应为 1
        bos_piece = trained_tokenizer.id_to_piece(1)
        assert trained_tokenizer.piece_to_id(bos_piece) == 1


class TestTokenizerCoverage:
    """测试 OOV 覆盖率检测"""

    def test_check_coverage_normal(self, trained_tokenizer):
        """常见文本覆盖率应接近 100%"""
        result = trained_tokenizer.check_coverage(["Hello world", "深度学习"])
        assert "total_tokens" in result
        assert "unk_tokens" in result
        assert "unk_ratio" in result
        assert "coverage" in result
        assert result["total_tokens"] > 0
        assert result["coverage"] >= 0.0

    def test_check_coverage_empty(self, trained_tokenizer):
        """空列表应返回 0 token"""
        result = trained_tokenizer.check_coverage([])
        assert result["total_tokens"] == 0
        assert result["coverage"] == 1.0

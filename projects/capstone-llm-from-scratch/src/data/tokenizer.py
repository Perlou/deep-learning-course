"""
Tokenizer — BPE 分词器封装
=========================

基于 sentencepiece 的 BPE (Byte Pair Encoding) 分词器。
为 ClearMind 提供统一的分词接口，支持中英文混合文本。

BPE 算法简述:
  1. 初始化: 将所有文本拆分为字符 (字节)
  2. 统计: 找出最频繁出现的相邻字符对
  3. 合并: 将最频繁的对合并为新符号
  4. 重复: 直到词表达到目标大小

这样常见的词 ("the", "的") 会被编码为单个 token,
罕见词则被拆分为多个子词。

特殊 token:
  <s>   = BOS (Beginning of Sequence), id=1
  </s>  = EOS (End of Sequence), id=2
  <unk> = Unknown token, id=0
"""

import os
from pathlib import Path

try:
    import sentencepiece as spm
except ModuleNotFoundError as exc:
    if exc.name == "sentencepiece":
        raise ModuleNotFoundError(
            "缺少依赖 `sentencepiece`，请先运行 `pip install -r requirements.txt`。"
        ) from exc
    raise


class ClearMindTokenizer:
    """ClearMind 分词器

    封装 sentencepiece 模型，提供简洁的 encode/decode 接口。

    Args:
        model_path: sentencepiece 模型文件路径 (.model)
    """

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        # 特殊 token ID
        self.bos_id = self.sp.bos_id()  # <s>  = 1
        self.eos_id = self.sp.eos_id()  # </s> = 2
        self.unk_id = self.sp.unk_id()  # <unk> = 0
        self.pad_id = self.sp.pad_id()  # <pad> = -1 (如果不存在)

        # 如果 pad_id 不存在, 使用 eos_id 作为 pad
        if self.pad_id == -1:
            self.pad_id = self.eos_id

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """将文本编码为 token ID 序列

        Args:
            text:    输入文本
            add_bos: 是否在开头添加 <s>
            add_eos: 是否在结尾添加 </s>

        Returns:
            token ID 列表
        """
        ids = self.sp.Encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码为文本

        Args:
            ids: token ID 列表

        Returns:
            解码后的文本
        """
        return self.sp.Decode(ids)

    def tokenize(self, text: str) -> list[str]:
        """将文本分词为 token 字符串列表 (用于调试)

        Args:
            text: 输入文本

        Returns:
            token 字符串列表
        """
        return self.sp.EncodeAsPieces(text)

    def id_to_piece(self, token_id: int) -> str:
        """将 token ID 转为 token 字符串"""
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """将 token 字符串转为 token ID"""
        return self.sp.PieceToId(piece)

    def check_coverage(self, texts: list[str]) -> dict:
        """检测文本的 OOV (未知 token) 覆盖率

        对给定文本列表进行编码，统计 <unk> token 的占比。
        当 unk 比率 > 5% 时打印警告。

        Args:
            texts: 待检测的文本列表

        Returns:
            dict with:
              total_tokens: 总 token 数
              unk_tokens:   unk token 数
              unk_ratio:    unk 占比 (0.0 ~ 1.0)
              coverage:     覆盖率 (1.0 - unk_ratio)
        """
        total_tokens = 0
        unk_tokens = 0

        for text in texts:
            ids = self.encode(text)
            total_tokens += len(ids)
            unk_tokens += sum(1 for i in ids if i == self.unk_id)

        if total_tokens == 0:
            return {
                "total_tokens": 0,
                "unk_tokens": 0,
                "unk_ratio": 0.0,
                "coverage": 1.0,
            }

        unk_ratio = unk_tokens / total_tokens
        coverage = 1.0 - unk_ratio

        if unk_ratio > 0.05:
            print(
                f"⚠️  OOV 警告: unk 比率 {unk_ratio:.2%} (共 {unk_tokens}/{total_tokens} tokens)"
            )
            print("   建议: 增大词表大小或增加训练数据覆盖率")

        return {
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "unk_ratio": unk_ratio,
            "coverage": coverage,
        }

    @staticmethod
    def train(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ):
        """训练分词器

        Args:
            input_file:         训练文本文件路径
            model_prefix:       输出模型文件前缀 (会生成 .model 和 .vocab 文件)
            vocab_size:         目标词表大小
            character_coverage: 字符覆盖率 (对中文等字符集大的语言需要高覆盖率)
            model_type:         模型类型 ("bpe" 或 "unigram")
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(model_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            # 特殊 token 配置
            pad_id=3,  # <pad> = 3
            bos_id=1,  # <s>   = 1
            eos_id=2,  # </s>  = 2
            unk_id=0,  # <unk> = 0
            # 训练参数
            num_threads=4,
            byte_fallback=True,  # 处理未见过的字符
            split_digits=True,  # 数字按位切分
            allow_whitespace_only_pieces=True,
        )

        print(f"✅ 分词器训练完成!")
        print(f"   模型: {model_prefix}.model")
        print(f"   词表: {model_prefix}.vocab")
        print(f"   词表大小: {vocab_size}")


if __name__ == "__main__":
    # 快速测试 (需要先训练分词器)
    import sys

    model_path = "outputs/tokenizer/tokenizer.model"
    if not os.path.exists(model_path):
        print(f"分词器模型不存在: {model_path}")
        print("请先运行 scripts/train_tokenizer.py 训练分词器")
        sys.exit(0)

    tokenizer = ClearMindTokenizer(model_path)
    print(f"词表大小: {tokenizer.vocab_size}")

    test_texts = [
        "Hello, world!",
        "深度学习是人工智能的核心技术。",
        "Transformer architecture has changed NLP. 注意力机制非常重要。",
    ]

    for text in test_texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        pieces = tokenizer.tokenize(text)
        decoded = tokenizer.decode(ids)
        print(f"\n原文: {text}")
        print(f"Tokens: {pieces}")
        print(f"IDs: {ids}")
        print(f"解码: {decoded}")

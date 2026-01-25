"""
分词器深入解析
==============

学习目标：
    1. 理解BPE、WordPiece等分词算法原理
    2. 使用Hugging Face Tokenizers库
    3. 比较不同分词器的效率
    4. 了解中文分词的特殊处理

核心概念：
    - BPE (Byte Pair Encoding): 字节对编码
    - WordPiece: BERT使用的分词算法
    - SentencePiece: 无监督分词
    - Token效率: 影响推理成本

前置知识：
    - Phase 11: NLP文本预处理
    - 基本的Python字符串操作
"""

import torch
import numpy as np
from collections import Counter, defaultdict
import re


# ==================== 第一部分：分词器概述 ====================


def introduction():
    """分词器介绍"""
    print("=" * 60)
    print("第一部分：分词器概述")
    print("=" * 60)

    print("""
分词器（Tokenizer）的作用：

文本 ←→ Token IDs ←→ 模型处理

┌─────────────────────────────────────────────────────────────┐
│                    分词流程                                  │
│                                                              │
│  "Hello, World!"                                             │
│         ↓  分词                                              │
│  ["Hello", ",", "World", "!"]                               │
│         ↓  转ID                                              │
│  [15496, 11, 2159, 0]                                       │
│         ↓  模型处理                                          │
│  [embedding vectors...]                                      │
└─────────────────────────────────────────────────────────────┘

为什么分词很重要？
1. 词表大小：影响模型参数量和计算量
2. Token效率：同样文本产生的token数量
3. OOV处理：如何处理未知词
4. 多语言支持：对不同语言的处理能力
    """)


# ==================== 第二部分：BPE算法 ====================


class SimpleBPE:
    """
    简化版BPE (Byte Pair Encoding) 实现

    核心思想：
    1. 从字符级别开始
    2. 反复合并最频繁的相邻token对
    3. 直到达到目标词表大小
    """

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.merges = {}  # 合并规则
        self.vocab = {}  # 词表

    def _get_pairs(self, word_freqs: dict) -> dict:
        """统计所有相邻token对的频率"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        """合并指定的token对"""
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def train(self, text: str):
        """训练BPE"""
        # 统计词频（每个词结尾加上特殊标记</w>）
        words = text.split()
        word_counts = Counter(words)

        # 初始化：将每个词拆分为字符
        word_freqs = {}
        for word, count in word_counts.items():
            # 添加结束标记
            chars = " ".join(list(word)) + " </w>"
            word_freqs[chars] = count

        # 初始词表：所有字符
        self.vocab = set()
        for word in word_freqs:
            for char in word.split():
                self.vocab.add(char)

        print(f"初始词表大小: {len(self.vocab)}")

        # 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break

            # 找最频繁的pair
            best_pair = max(pairs, key=pairs.get)

            # 合并
            word_freqs = self._merge_pair(best_pair, word_freqs)

            # 记录合并规则
            self.merges[best_pair] = "".join(best_pair)
            self.vocab.add("".join(best_pair))

            if (i + 1) % 10 == 0:
                print(f"合并 {i + 1}: {best_pair} -> {''.join(best_pair)}")

        print(f"最终词表大小: {len(self.vocab)}")

    def tokenize(self, text: str) -> list:
        """使用训练好的BPE分词"""
        words = text.split()
        tokens = []

        for word in words:
            # 拆分为字符
            word_tokens = list(word) + ["</w>"]

            # 应用合并规则
            while True:
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                # 找第一个可合并的pair
                merge_pair = None
                for pair in pairs:
                    if pair in self.merges:
                        merge_pair = pair
                        break

                if merge_pair is None:
                    break

                # 执行合并
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (
                        i < len(word_tokens) - 1
                        and (word_tokens[i], word_tokens[i + 1]) == merge_pair
                    ):
                        new_tokens.append(self.merges[merge_pair])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            tokens.extend(word_tokens)

        return tokens


def bpe_examples():
    """BPE示例"""
    print("\n" + "=" * 60)
    print("第二部分：BPE (Byte Pair Encoding)")
    print("=" * 60)

    print("\nBPE算法步骤:")
    print("1. 初始词表 = 所有字符")
    print("2. 统计相邻token对的频率")
    print("3. 合并最高频的pair，加入词表")
    print("4. 重复直到达到目标词表大小")

    # 简单示例
    text = "low lower lowest new newer newest"
    print(f"\n训练文本: '{text}'")

    bpe = SimpleBPE(vocab_size=30)
    bpe.train(text)

    print("\n合并规则:")
    for pair, merged in list(bpe.merges.items())[:5]:
        print(f"  {pair} -> {merged}")

    # 测试分词
    test_text = "low newest"
    tokens = bpe.tokenize(test_text)
    print(f"\n测试分词 '{test_text}':")
    print(f"  Tokens: {tokens}")


# ==================== 第三部分：使用Hugging Face Tokenizers ====================


def huggingface_tokenizer_examples():
    """Hugging Face Tokenizers示例"""
    print("\n" + "=" * 60)
    print("第三部分：Hugging Face Tokenizers")
    print("=" * 60)

    print("""
安装: pip install transformers tokenizers

常用Tokenizer类型：
┌─────────────────────────────────────────────────────────────┐
│  类型          │  模型         │  特点                      │
├─────────────────────────────────────────────────────────────┤
│  BPE           │  GPT-2, GPT-4 │  字节级BPE，处理任意文本   │
│  WordPiece     │  BERT         │  ##前缀标记子词           │
│  SentencePiece │  LLaMA, T5    │  语言无关，直接处理原始文本│
│  Tiktoken      │  OpenAI       │  高效，Rust实现           │
└─────────────────────────────────────────────────────────────┘
    """)

    code_example = """
from transformers import AutoTokenizer

# 加载不同模型的tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 基本使用
text = "Hello, how are you today?"

# 编码
tokens = gpt2_tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

token_ids = gpt2_tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# 解码
decoded = gpt2_tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# 批量编码（自动padding）
texts = ["Hello world", "How are you doing today?"]
encoded = gpt2_tokenizer(texts, padding=True, return_tensors="pt")
print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Attention Mask: {encoded['attention_mask']}")

# 特殊token
print(f"PAD token: {gpt2_tokenizer.pad_token}")
print(f"EOS token: {gpt2_tokenizer.eos_token}")
print(f"词表大小: {len(gpt2_tokenizer)}")
"""
    print(code_example)


# ==================== 第四部分：Token效率对比 ====================


def token_efficiency_comparison():
    """Token效率对比"""
    print("\n" + "=" * 60)
    print("第四部分：Token效率对比")
    print("=" * 60)

    print("""
不同Tokenizer的效率差异很大，尤其对于中文：

┌─────────────────────────────────────────────────────────────┐
│                    Token效率对比示例                         │
│                                                              │
│  文本: "深度学习改变了自然语言处理的发展"                      │
│                                                              │
│  Tokenizer           词表大小    Token数    效率             │
│  ────────────────────────────────────────────────           │
│  GPT-2 (50257)       50K        ~25       低               │
│  LLaMA (32000)       32K        ~15       中               │
│  Qwen (151643)       152K       ~10       高               │
│  GLM (130528)        131K       ~11       高               │
│                                                              │
│  效率 = 文本字符数 / Token数                                 │
│  更高效 = 更低的推理成本 + 更长的有效上下文                   │
└─────────────────────────────────────────────────────────────┘

为什么Qwen对中文更高效？
1. 训练数据包含大量中文
2. 词表中有更多中文词汇
3. 常见中文词作为单一token
    """)

    code_example = """
# 比较不同tokenizer的效率
from transformers import AutoTokenizer

def compare_tokenizers(text, tokenizer_names):
    print(f"文本: {text}")
    print(f"字符数: {len(text)}")
    print("-" * 50)
    
    for name in tokenizer_names:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            tokens = tokenizer.encode(text)
            print(f"{name}: {len(tokens)} tokens")
        except Exception as e:
            print(f"{name}: 加载失败")

text = "深度学习是人工智能的核心技术，正在改变世界"
compare_tokenizers(text, [
    "gpt2",
    "Qwen/Qwen2.5-7B-Instruct",
    "THUDM/glm-4-9b"
])
"""
    print(code_example)


# ==================== 第五部分：训练自定义Tokenizer ====================


def custom_tokenizer_training():
    """训练自定义Tokenizer"""
    print("\n" + "=" * 60)
    print("第五部分：训练自定义Tokenizer")
    print("=" * 60)

    code_example = """
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 创建BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 设置预分词器
tokenizer.pre_tokenizer = Whitespace()

# 3. 定义训练器
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. 训练（从文件或字符串）
# 方式1: 从文件训练
files = ["data/corpus.txt"]
tokenizer.train(files, trainer)

# 方式2: 从字符串训练
tokenizer.train_from_iterator(
    ["Hello world", "How are you"],
    trainer
)

# 5. 保存和加载
tokenizer.save("my_tokenizer.json")
loaded_tokenizer = Tokenizer.from_file("my_tokenizer.json")

# 6. 使用
output = tokenizer.encode("Hello, how are you?")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
"""
    print(code_example)

    print("""
训练自定义Tokenizer的场景：
1. 特定领域语料（医学、法律）
2. 特殊语言（方言、古文）
3. 代码和符号密集的文本
4. 追求更高的token效率
    """)


# ==================== 第六部分：特殊Token处理 ====================


def special_tokens_handling():
    """特殊Token处理"""
    print("\n" + "=" * 60)
    print("第六部分：特殊Token处理")
    print("=" * 60)

    print("""
LLM中的特殊Token：

┌─────────────────────────────────────────────────────────────┐
│  Token         │  作用                     │  示例          │
├─────────────────────────────────────────────────────────────┤
│  [BOS]         │  句子开始                  │  <s>, <|im_start|> │
│  [EOS]         │  句子结束                  │  </s>, <|im_end|>  │
│  [PAD]         │  填充                      │  <pad>            │
│  [UNK]         │  未知词                    │  <unk>            │
│  [SEP]         │  分隔符                    │  <sep>            │
│  [MASK]        │  掩码（BERT用）            │  [MASK]           │
└─────────────────────────────────────────────────────────────┘

对话模板中的特殊Token：

ChatML格式 (OpenAI):
<|im_start|>system
你是一个有帮助的AI助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的？<|im_end|>

LLaMA格式:
[INST] <<SYS>>
你是一个有帮助的AI助手
<</SYS>>

用户问题 [/INST] 助手回复
    """)

    code_example = """
from transformers import AutoTokenizer

# 查看特殊token
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

print("特殊Token:")
print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 应用对话模板
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手"},
    {"role": "user", "content": "你好"}
]

# 使用内置的chat_template
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"\\n对话模板结果:\\n{text}")
"""
    print(code_example)


# ==================== 第七部分：练习 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现WordPiece分词
    任务：实现简化版WordPiece算法
    提示：与BPE的区别在于使用似然增益而非频率

练习 1 答案：
    class SimpleWordPiece:
        def tokenize(self, word, vocab):
            '''从左到右贪婪匹配最长子词'''
            tokens = []
            start = 0
            while start < len(word):
                end = len(word)
                found = None
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in vocab:
                        found = substr
                        break
                    end -= 1
                if found:
                    tokens.append(found)
                    start = end
                else:
                    tokens.append("[UNK]")
                    start += 1
            return tokens

练习 2：计算Token效率
    任务：比较不同文本类型的token效率
    文本类型：英文、中文、代码、数学公式

练习 2 答案：
    def calculate_efficiency(tokenizer, text):
        tokens = tokenizer.encode(text)
        chars = len(text)
        efficiency = chars / len(tokens)
        return efficiency
    
    # 预期结果：
    # 英文效率最高（1 token ≈ 4-5字符）
    # 中文效率依tokenizer而异
    # 代码和数学符号效率通常较低

思考题：为什么需要子词分词？
    答案：
    1. 平衡词表大小和覆盖率
    2. 处理未登录词（OOV）
    3. 共享词素信息（如 "run", "running" 共享 "run"）
    4. 支持任意输入（理论上不会有UNK）
    """)


# ==================== 主函数 ====================


def main():
    """主函数 - 按顺序执行所有部分"""
    introduction()
    bpe_examples()
    huggingface_tokenizer_examples()
    token_efficiency_comparison()
    custom_tokenizer_training()
    special_tokens_handling()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！")
    print("=" * 60)
    print("""
下一步学习：
    - 03-flash-attention.py: Flash Attention原理
    
关键要点回顾：
    ✓ BPE通过合并最频繁的token对构建词表
    ✓ WordPiece使用似然增益选择合并
    ✓ Token效率影响推理成本和上下文长度
    ✓ 中文优化的tokenizer对中文更高效
    ✓ 特殊token在对话模板中很重要
    """)


if __name__ == "__main__":
    main()

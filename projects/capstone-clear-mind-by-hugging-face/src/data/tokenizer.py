"""
tokenizer.py — ClearMind Tokenizer (HF tokenizers 版)
=====================================================

使用 HuggingFace tokenizers 库训练 ByteLevel BPE tokenizer，
包装为 PreTrainedTokenizerFast 支持 save_pretrained/from_pretrained。

与 from-scratch 版对比:
  - from-scratch: sentencepiece → .model 文件，手写 encode/decode
  - HF 版: tokenizers BPE → tokenizer.json，PreTrainedTokenizerFast 自动处理
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# Human/Assistant 对话格式 chat template (Jinja2)
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "Human: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "Assistant: "
    "{% endif %}"
)


class ClearMindTokenizer:
    """ClearMind tokenizer 工厂类 — 训练和加载 BPE tokenizer"""

    # 特殊 token 定义
    SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"

    @staticmethod
    def train(
        corpus_path: str,
        vocab_size: int = 2000,
        character_coverage: float = 0.9995,
    ) -> PreTrainedTokenizerFast:
        """训练 ByteLevel BPE tokenizer 并包装为 PreTrainedTokenizerFast

        Args:
            corpus_path: 训练语料文件路径 (纯文本，每行一条)
            vocab_size: 词表大小
            character_coverage: 字符覆盖率 (影响 BPE 训练，用于 trainer 参数参考)

        Returns:
            PreTrainedTokenizerFast: 可直接用于 HF 生态的 tokenizer
        """
        # 1. 构建底层 tokenizer
        tokenizer = Tokenizer(BPE(unk_token=ClearMindTokenizer.UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        # 2. 训练 BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=ClearMindTokenizer.SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train([corpus_path], trainer=trainer)

        # 3. 配置 post processor — 自动添加 BOS/EOS
        bos_id = tokenizer.token_to_id(ClearMindTokenizer.BOS_TOKEN)
        eos_id = tokenizer.token_to_id(ClearMindTokenizer.EOS_TOKEN)
        tokenizer.post_processor = TemplateProcessing(
            single=f"<s>:0 $A:0 </s>:0",
            pair=f"<s>:0 $A:0 </s>:0 <s>:1 $B:1 </s>:1",
            special_tokens=[
                ("<s>", bos_id),
                ("</s>", eos_id),
            ],
        )

        # 4. 包装为 PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token=ClearMindTokenizer.UNK_TOKEN,
            bos_token=ClearMindTokenizer.BOS_TOKEN,
            eos_token=ClearMindTokenizer.EOS_TOKEN,
            pad_token=ClearMindTokenizer.PAD_TOKEN,
            chat_template=CHAT_TEMPLATE,
        )

        return fast_tokenizer

    @staticmethod
    def load(path: str) -> PreTrainedTokenizerFast:
        """从目录加载已保存的 tokenizer

        Args:
            path: tokenizer 保存目录 (包含 tokenizer.json 等文件)

        Returns:
            PreTrainedTokenizerFast
        """
        return AutoTokenizer.from_pretrained(path)

"""
HFTokenizer — HuggingFace tokenizer 包装层
==========================================

对外暴露与 :class:`ClearMindTokenizer` 一致的接口（vocab_size / encode / decode /
bos_id / eos_id / pad_id / unk_id），同时新增以下生产环境能力：

  - ``apply_chat_template`` 直接渲染多轮对话（system / user / assistant / tool）
  - ``encode_with_offsets`` 返回 token id + 是否 special 的标记，便于 SFT loss mask
  - ``find_subseq`` 在 token 序列中扫描标记子序列（用于定位 ``<|im_start|>assistant\\n``）
  - ``save_pretrained`` 透传到底层 tokenizer，便于发布到 HuggingFace / ModelScope

为什么要用 HF AutoTokenizer 而不是自家 sentencepiece 包装？
  ClearMind 的目标是发布生产级模型到 HuggingFace / ModelScope，并被 ollama / vllm /
  llama.cpp / Llama-Factory 等生态消费。HF tokenizer.json + tokenizer_config.json
  是这条链路的事实标准（包含 chat_template、special tokens、added_tokens 等）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        raise ModuleNotFoundError(
            "缺少依赖 `transformers`，请先运行 `pip install -r requirements.txt`。"
        ) from exc
    raise


# 仓库内置 tokenizer 路径（复制自 minimind/model/，独立于 data/，便于 git 追踪）
DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[2] / "tokenizer" / "minimind"
)


class HFTokenizer:
    """HuggingFace tokenizer 的轻量包装

    设计目标：
      1. 与 :class:`ClearMindTokenizer` 接口对齐，让 dataset / trainer 代码无需改动
         即可在 sentencepiece 与 HF tokenizer 之间切换
      2. 暴露 chat_template 与 mark-token 扫描能力，用于 SFT/DPO 的 loss mask 生成
      3. 支持 ``save_pretrained`` 与发布生态打通

    Args:
        model_path: HF tokenizer 目录（包含 tokenizer.json + tokenizer_config.json）
                    或 HuggingFace Hub repo id。默认指向 ``tokenizer/minimind``。
        trust_remote_code: 是否信任 tokenizer 仓库内的自定义代码（一般不需要）。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        trust_remote_code: bool = False,
    ):
        path = Path(model_path) if model_path else DEFAULT_TOKENIZER_DIR
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            str(path),
            trust_remote_code=trust_remote_code,
        )
        self._path = str(path)

        # 与 ClearMindTokenizer 对齐的特殊 token id
        self.bos_id: int = self._tokenizer.bos_token_id
        self.eos_id: int = self._tokenizer.eos_token_id
        self.pad_id: int = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else self.eos_id
        )
        self.unk_id: int = (
            self._tokenizer.unk_token_id
            if self._tokenizer.unk_token_id is not None
            else self.pad_id
        )

        # 缓存常用 mark token 序列（loss mask 扫描用）
        # bos_assistant: <|im_start|>assistant\n  → 标记 assistant 段开始
        # eos_newline:   <|im_end|>\n            → 标记一段结束
        bos_token = self._tokenizer.bos_token or ""
        eos_token = self._tokenizer.eos_token or ""
        self.assistant_prefix_ids: list[int] = self._tokenizer(
            f"{bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_with_newline_ids: list[int] = self._tokenizer(
            f"{eos_token}\n", add_special_tokens=False
        ).input_ids

    # ----------------------------------------------------------
    # 与 ClearMindTokenizer 一致的接口
    # ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """完整词表大小（含 added_tokens，便于 ModelConfig.vocab_size 对齐）"""
        return len(self._tokenizer)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """将文本编码为 token id 列表

        ``add_bos`` / ``add_eos`` 显式控制是否包裹 BOS/EOS，避免 HF tokenizer 默认
        ``add_special_tokens=True`` 在不同 chat template 下行为不一致带来的 bug。
        """
        ids = self._tokenizer(text, add_special_tokens=False).input_ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """将 token id 列表解码回文本"""
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> list[str]:
        """返回 token 字符串列表（调试用）"""
        return self._tokenizer.tokenize(text)

    def id_to_piece(self, token_id: int) -> str:
        return self._tokenizer.convert_ids_to_tokens(int(token_id))

    def piece_to_id(self, piece: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(piece)

    # ----------------------------------------------------------
    # chat_template / 多轮对话渲染
    # ----------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **template_kwargs,
    ) -> str | list[int]:
        """渲染多轮对话为字符串或 token 序列

        ``messages`` 接受 minimind 风格的 ``[{"role": "...", "content": "..."}, ...]``
        列表，支持 ``role in {"system", "user", "assistant", "tool"}``，以及 assistant
        消息中的 ``reasoning_content`` 与 ``tool_calls`` 字段。

        Args:
            messages: 对话消息列表
            tools: 可选的工具定义列表（OpenAI tool schema），用于 tool-use 数据
            add_generation_prompt: 是否在末尾追加 ``<|im_start|>assistant\\n``，
                推理时设 True，训练时设 False
            tokenize: True 时返回 token id 列表，False 时返回字符串
            template_kwargs: 透传给 jinja 模板的额外变量（例如 ``open_thinking=True``）

        Note:
            为兼容不同版本 transformers 对 ``tokenize=True`` 的返回类型差异
            （新版返回 ``BatchEncoding`` dict，旧版返回 list），统一走"先渲染
            字符串、再 ``encode`` 拿 ids"的路径。
        """
        text: str = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **template_kwargs,
        )
        if not tokenize:
            return text
        return self._tokenizer(text, add_special_tokens=False).input_ids

    # ----------------------------------------------------------
    # mark-token 扫描（SFT/DPO loss mask）
    # ----------------------------------------------------------

    @staticmethod
    def find_subseq(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
        """在 ``haystack`` 中找 ``needle`` 出现的所有起始下标

        用于定位 ``<|im_start|>assistant\\n`` 等 mark 序列在 token 流中的位置，
        进而划定 SFT loss mask 的"只对 assistant 段计算 loss"区间。
        """
        if not needle or len(needle) > len(haystack):
            return []
        n, m = len(haystack), len(needle)
        out: list[int] = []
        for i in range(n - m + 1):
            if all(haystack[i + j] == needle[j] for j in range(m)):
                out.append(i)
        return out

    def generate_assistant_mask(self, input_ids: Sequence[int]) -> list[int]:
        """根据 token 流扫描出 assistant 段，返回 0/1 loss mask（与 input_ids 等长）

        实现策略对齐 minimind ``SFTDataset.generate_labels``：
          - 每发现一段 ``assistant_prefix_ids`` 就开始填 1
          - 直到下一个 ``eos_with_newline_ids`` 为止（含 eos 行）
          - 其它位置（system/user/tool）保持 0

        Returns:
            mask: 长度等于 ``len(input_ids)`` 的 0/1 list
        """
        ids = list(input_ids)
        n = len(ids)
        mask = [0] * n
        prefix = self.assistant_prefix_ids
        suffix = self.eos_with_newline_ids
        plen, slen = len(prefix), len(suffix)
        i = 0
        while i < n:
            # 匹配 assistant 段开始
            if i + plen <= n and ids[i : i + plen] == prefix:
                start = i + plen
                end = start
                while end < n:
                    if end + slen <= n and ids[end : end + slen] == suffix:
                        break
                    end += 1
                # 把 assistant 内容 + eos 都标为 1（学完整段，包括 EOS 让模型学会停）
                stop = min(end + slen, n)
                for j in range(start, stop):
                    mask[j] = 1
                i = stop
            else:
                i += 1
        return mask

    # ----------------------------------------------------------
    # 持久化（发布到 HF / ModelScope 用）
    # ----------------------------------------------------------

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> list[str]:
        """透传到底层 HF tokenizer 的 save_pretrained，发布生态打通"""
        return self._tokenizer.save_pretrained(str(save_directory), **kwargs)

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """暴露底层 PreTrainedTokenizerBase，需要时直接调用 HF API"""
        return self._tokenizer

    def __repr__(self) -> str:
        return (
            f"HFTokenizer(path={self._path!r}, vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}, pad={self.pad_id})"
        )


if __name__ == "__main__":
    tok = HFTokenizer()
    print(tok)
    print("assistant_prefix_ids:", tok.assistant_prefix_ids)
    print("eos_with_newline_ids:", tok.eos_with_newline_ids)

    msgs = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
    ]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    print("\n--- rendered ---")
    print(rendered)

    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    mask = tok.generate_assistant_mask(ids)
    print("\n--- mask preview ---")
    print(f"len(ids)={len(ids)}, sum(mask)={sum(mask)}")
    # 打印 mask=1 的 token 文本，验证只覆盖 assistant 段
    masked_ids = [i for i, m in zip(ids, mask) if m]
    print("masked decode:", tok.decode(masked_ids))

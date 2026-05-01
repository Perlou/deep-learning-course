"""
SFTDataset — 指令微调数据集
===========================

支持两种数据格式：

1. **conversations（minimind / ShareGPT 格式，默认）**：
   每行 ``{"conversations": [{"role": "system|user|assistant|tool", "content": "...",
   "reasoning_content": "...", "tools": "...", "tool_calls": [...]}]}``

2. **Alpaca（向后兼容）**：
   每行 ``{"instruction": "...", "input": "...", "output": "..."}``，自动转换为
   ``conversations=[{"role":"user","content":...}, {"role":"assistant","content":...}]``

Loss mask 由 :meth:`HFTokenizer.generate_assistant_mask` 通过扫描
``<|im_start|>assistant\\n`` 与 ``<|im_end|>\\n`` 的 token 序列生成，避免子串截取在
BPE 边界处错位的问题（这是 minimind 与原 ClearMind SFTDataset 的关键差异之一）。

数据增强（与 minimind 行为对齐）：
  - ``system_prompt_ratio``：当样本没有 system 消息时，按概率插入一段随机 system prompt
  - ``empty_think_strip_ratio``：按概率移除渲染文本中的空 ``<think>\\n\\n</think>\\n\\n``，
    避免模型把"先输出空思考"当作硬规则
"""

from __future__ import annotations

import json
import os
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


# 默认的 system prompt 池，用于数据增强（中英混合，覆盖不同语境）
DEFAULT_SYSTEM_PROMPTS: list[str] = [
    "你是一个知识丰富的AI助手，请尽力为用户提供准确的回答。",
    "你是 ClearMind，一个轻量但有用的语言模型。",
    "你是一个专业的中文AI助手，请提供有价值的回答。",
    "你是一个可靠的AI，请给出准确清晰的回答。",
    "You are a helpful AI assistant.",
    "You are ClearMind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
]


def _normalize_message(msg: dict) -> dict:
    """规整一条 message：兼容 ShareGPT 旧字段 + minimind tool_calls 字符串

    特别处理（参考 minimind/dataset/lm_dataset.py）：
      - ``tool_calls`` 字段可能是 JSON 字符串 → ``json.loads`` 转 list
      - 每个 tool_call 必须含 ``name`` 与 ``arguments``，否则 jinja 模板会触发
        ``Undefined.tojson()`` 报 ``TypeError``
      - ``tools`` 字段（system 消息上）单独由 :func:`_extract_conversations` 提取，
        这里不放进 message 字典
    """
    if "role" in msg:
        out = {"role": msg["role"], "content": msg.get("content", "")}
    else:
        # ShareGPT: {"from": "human"/"gpt", "value": "..."}
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
            "tool": "tool",
            "observation": "tool",
        }
        out = {
            "role": role_map.get(msg.get("from", "user"), "user"),
            "content": msg.get("value", ""),
        }

    # 透传 reasoning_content（assistant 思考链）
    if msg.get("reasoning_content") is not None:
        out["reasoning_content"] = msg["reasoning_content"]

    # tool_calls 处理（minimind 风格）：字符串 → list；过滤格式不完整的
    tool_calls = msg.get("tool_calls")
    if tool_calls is not None and tool_calls != "":
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                tool_calls = None
        if isinstance(tool_calls, list):
            cleaned: list[dict] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                # 兼容 OpenAI 风格：{"function": {"name":..., "arguments":...}}
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                if "name" not in fn or "arguments" not in fn:
                    continue
                # 把 arguments 序列化为字符串（jinja 模板的 tojson 兜底依赖这个）
                args = fn.get("arguments")
                if not isinstance(args, str):
                    try:
                        fn["arguments"] = json.dumps(args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        continue
                cleaned.append(tc)
            if cleaned:
                out["tool_calls"] = cleaned

    return out


def _alpaca_to_conversations(sample: dict) -> list[dict]:
    """Alpaca → conversations 转换"""
    instruction = sample.get("instruction", "")
    user_input = sample.get("input", "")
    output = sample.get("output", "")
    user_text = f"{instruction}\n{user_input}".strip() if user_input else instruction
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": output},
    ]


def _extract_conversations(sample: dict) -> tuple[list[dict], list[dict] | None]:
    """从样本中提取 (conversations, tools)

    Returns:
        (messages, tools)：tools 来自 system 消息的 ``tools`` 字段（已 json.loads）。
        没有 tools 字段时 tools=None。
    """
    if "conversations" in sample and sample["conversations"]:
        convs: list[dict] = []
        tools: list[dict] | None = None
        for raw in sample["conversations"]:
            # system 消息的 tools 字段单独提取（minimind 风格，不放回 message）
            if (
                isinstance(raw, dict)
                and raw.get("role") == "system"
                and raw.get("tools")
            ):
                t = raw["tools"]
                if isinstance(t, str):
                    try:
                        tools = json.loads(t)
                    except json.JSONDecodeError:
                        tools = None
                elif isinstance(t, list):
                    tools = t
            convs.append(_normalize_message(raw))
        return convs, tools
    if "messages" in sample and sample["messages"]:
        return [_normalize_message(m) for m in sample["messages"]], None
    if "instruction" in sample:
        return _alpaca_to_conversations(sample), None
    raise ValueError(
        f"无法识别的 SFT 样本格式，可用 keys: {list(sample.keys())}"
    )


class SFTDataset(Dataset):
    """SFT 指令微调数据集

    Args:
        data_path:                   ``.jsonl`` 或 ``.json`` 数据文件路径
        tokenizer:                   :class:`HFTokenizer` 实例（必须支持 chat_template）
        max_seq_len:                 最大序列长度
        system_prompt_ratio:         无 system 消息时注入随机 system 的概率，0 关闭
        empty_think_strip_ratio:     移除空 ``<think>\\n\\n</think>\\n\\n`` 的概率，0 关闭
        system_prompt_pool:          system prompt 候选池（None 用默认）
        seed:                        数据增强随机种子（None 时不固定）
        _samples:                    内部使用——直接传入样本列表
    """

    def __init__(
        self,
        data_path: str | None = None,
        tokenizer=None,
        max_seq_len: int = 1024,
        system_prompt_ratio: float = 0.2,
        empty_think_strip_ratio: float = 0.8,
        system_prompt_pool: list[str] | None = None,
        seed: int | None = None,
        _samples: list[dict] | None = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("SFTDataset 需要 tokenizer（HFTokenizer 实例）")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt_ratio = system_prompt_ratio
        self.empty_think_strip_ratio = empty_think_strip_ratio
        self.system_prompt_pool = list(system_prompt_pool or DEFAULT_SYSTEM_PROMPTS)
        # 每个 dataset 实例独立的 RNG，避免全局污染
        self._rng = random.Random(seed)

        if _samples is not None:
            self.samples: list[dict] = list(_samples)
        else:
            print(f"📦 加载 SFT 数据: {data_path}")
            self.samples = self._load_data(data_path)
        print(f"  样本数: {len(self.samples):,}")

    # ----------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------

    @staticmethod
    def _load_data(data_path: str) -> list[dict]:
        ext = os.path.splitext(data_path)[1]
        if ext == ".jsonl":
            out: list[dict] = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
            return out
        if ext == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"不支持的文件格式: {ext}")

    @classmethod
    def create_with_split(
        cls,
        data_path: str,
        tokenizer,
        max_seq_len: int = 1024,
        val_ratio: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> tuple["SFTDataset", "SFTDataset"]:
        """创建训练/验证集（按样本数切分）"""
        with open(data_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        rng = random.Random(seed)
        rng.shuffle(samples)
        split = int(len(samples) * (1 - val_ratio))
        print(f"📊 SFT 数据划分: 训练 {split:,} / 验证 {len(samples) - split:,}")
        train = cls(
            _samples=samples[:split],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            seed=seed,
            **kwargs,
        )
        val = cls(
            _samples=samples[split:],
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            seed=seed + 1,
            # 验证集关闭随机增强，结果稳定
            system_prompt_ratio=0.0,
            empty_think_strip_ratio=0.0,
            **{k: v for k, v in kwargs.items()
               if k not in ("system_prompt_ratio", "empty_think_strip_ratio")},
        )
        return train, val

    # ----------------------------------------------------------
    # 数据增强
    # ----------------------------------------------------------

    def _augment_conversations(self, convs: list[dict]) -> list[dict]:
        """概率性注入 system prompt（仅当原始对话无 system 时）"""
        if not convs or self.system_prompt_ratio <= 0:
            return convs
        if convs[0].get("role") == "system":
            return convs
        # tool-use 数据通常需要专用 system，不注入
        if any("tools" in m or "tool_calls" in m for m in convs):
            return convs
        if self._rng.random() < self.system_prompt_ratio:
            sys_msg = {
                "role": "system",
                "content": self._rng.choice(self.system_prompt_pool),
            }
            return [sys_msg] + convs
        return convs

    def _post_process(self, text: str) -> str:
        """概率性移除空 think 占位"""
        if self.empty_think_strip_ratio <= 0:
            return text
        marker = "<think>\n\n</think>\n\n"
        if marker in text and self._rng.random() < self.empty_think_strip_ratio:
            text = text.replace(marker, "")
        return text

    # ----------------------------------------------------------
    # Dataset 接口
    # ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    # ----------------------------------------------------------
    # 单样本编码：返回 None 表示"该样本无法产出有效监督信号"
    # （例如 user 单轮就超过 max_seq_len，截断后 assistant 段被砍光）
    # ----------------------------------------------------------
    def _encode_sample(self, sample: dict) -> dict | None:
        convs, tools = _extract_conversations(sample)
        convs = self._augment_conversations(convs)

        # 渲染为字符串 → 后处理 → tokenize
        # 注意：tools 必须显式传给 apply_chat_template（minimind chat_template
        # 中的 ``{%- if tools %}`` 分支依赖此参数；放在 message 字典中不会被识别）
        try:
            prompt = self.tokenizer.apply_chat_template(
                convs,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
        except (TypeError, ValueError):
            # chat_template 渲染失败（通常是某条 tool_calls 格式异常），
            # 跳过 tool_calls 字段重试，让训练能继续
            convs_safe = [
                {k: v for k, v in m.items() if k != "tool_calls"} for m in convs
            ]
            prompt = self.tokenizer.apply_chat_template(
                convs_safe,
                add_generation_prompt=False,
                tokenize=False,
            )
        prompt = self._post_process(prompt)

        ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)

        # 先在"完整 ids"上扫描 mask（保证 marker token 序列完整未被截断破坏），
        # 再统一对 ids/mask 做尾部截断（保留最后一轮 assistant，丢弃头部上下文）。
        # 这是 ClearMind 与 minimind 的差异之一：minimind 走头部截断容易把唯一一段
        # assistant 砍光 → CE(reduction='mean') 0/0 = NaN 污染整个 epoch。
        mask = self.tokenizer.generate_assistant_mask(ids)

        if len(ids) > self.max_seq_len:
            ids = ids[-self.max_seq_len:]
            mask = mask[-self.max_seq_len:]

        # 全 -100 兜底：极端长单轮 user / 截断后没有任何 assistant token 的样本
        # 直接 return None，由 __getitem__ 切换到下一个样本
        if sum(mask) == 0:
            return None

        # Padding
        pad_id = self.tokenizer.pad_id
        pad_len = self.max_seq_len - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        # labels: mask=1 处保留 token id，其它位置（包括 padding）置 -100
        labels = input_ids.clone()
        mask_t = torch.tensor(mask, dtype=torch.long)
        labels[mask_t == 0] = -100

        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, idx: int) -> dict:
        """获取单个样本

        防御策略（A 方案 Layer 1）：
          - 当前样本编码失败或全 -100 时，依次向后试 8 个样本
          - 8 次都失败时返回全 -100 兜底（极少触发；下游 loss 层会安全返回 0）
        """
        n = len(self.samples)
        max_retry = min(8, n)
        for offset in range(max_retry):
            sample = self.samples[(idx + offset) % n]
            try:
                result = self._encode_sample(sample)
            except Exception:
                # 单个样本编码异常不应终止整个训练
                continue
            if result is not None:
                return result

        # 全部失败兜底（理论极少触发）
        pad_id = self.tokenizer.pad_id
        L = self.max_seq_len
        return {
            "input_ids": torch.full((L,), pad_id, dtype=torch.long),
            "labels": torch.full((L,), -100, dtype=torch.long),
        }

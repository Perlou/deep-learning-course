"""
RL 数据集（占位）— RLAIFDataset / AgentRLDataset
=================================================

用于 PPO / GRPO / Agentic RL 训练的数据读取层。Trainer 实现属于
``src/training/{ppo,grpo,agent}.py``（计划中 Phase 3 实施），但数据加载层先就位，
方便数据下载后立即可用。

数据格式（与 minimind 对齐）：

- **RLAIF**：``{"conversations": [{"role": "user|system", "content": "..."}, ...]}``
  最后一条通常是 user 提问；trainer 会用 rollout 引擎生成 assistant 回复，再用
  reward model 打分。
- **Agent RL**：``{"conversations": [{"role": "user", "content": "..."}, ...,
  {"role": "assistant", "content": "..."}], "gt": "..."}``
  含 ground truth 答案，trainer 在多轮 tool-call rollout 后用 ``gt`` 做规则奖励。

这里只负责"读 + 渲染 prompt + 透传 gt/tools"，不构造 input_ids（rollout 阶段才需要）。
"""

from __future__ import annotations

import json
import os
import random

from torch.utils.data import Dataset


def _normalize_message(msg: dict) -> dict:
    out = dict(msg)
    if "role" not in out and "from" in out:
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
        out["role"] = role_map.get(out["from"], "user")
        out["content"] = out.get("value", "")
    return out


class RLAIFDataset(Dataset):
    """RLAIF (PPO/GRPO) 数据集

    每条样本输出:
      ``{"prompt": <已渲染好的 chat prompt 字符串>, "answer": "", "messages": <原始>}``

    Trainer 会用 ``prompt`` 做 rollout（生成 K 条 response），然后用 reward function
    或 reward model 打分计算 advantage。

    Args:
        data_path:        ``.jsonl`` 文件
        tokenizer:        :class:`HFTokenizer` 实例
        max_prompt_len:   prompt 最大 token 数（超长截断）
        thinking_ratio:   渲染时 ``open_thinking=True`` 的概率（控制是否开启思考链）
        seed:             随机种子
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_prompt_len: int = 1024,
        thinking_ratio: float = 0.5,
        seed: int | None = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("RLAIFDataset 需要 tokenizer")
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.thinking_ratio = thinking_ratio
        self._rng = random.Random(seed)

        print(f"📦 加载 RLAIF 数据: {data_path}")
        self.samples = self._load(data_path)
        print(f"  样本数: {len(self.samples):,}")

    @staticmethod
    def _load(data_path: str) -> list[dict]:
        if not data_path.endswith(".jsonl"):
            raise ValueError(f"RLAIFDataset 仅支持 .jsonl，收到 {data_path!r}")
        out: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        convs = [_normalize_message(m) for m in sample.get("conversations", [])]
        # 取除最后一条外的历史作为 prompt（最后一条若是 assistant 则丢弃，让 rollout 自己生成）
        if convs and convs[-1].get("role") == "assistant":
            convs = convs[:-1]

        open_thinking = self._rng.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=False,
            open_thinking=open_thinking,
        )
        return {"prompt": prompt, "answer": "", "messages": convs}


class AgentRLDataset(Dataset):
    """Agentic Tool-Use RL 数据集

    每条样本输出:
      ``{"messages": <原始多轮对话历史>, "tools": <工具定义列表>, "gt": <ground truth>}``

    Trainer（计划中的 Agent RL）会做多轮 tool-call rollout，用 ``gt`` 做规则奖励
    （比如答案中是否包含正确数值）。

    Args:
        data_path:    ``.jsonl`` 文件
        tokenizer:    :class:`HFTokenizer` 实例（dataset 本身不 tokenize，只是预留接口）
        max_seq_len:  单 turn 最大长度（trainer 内部使用）
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        print(f"📦 加载 Agent RL 数据: {data_path}")
        self.samples = self._load(data_path)
        print(f"  样本数: {len(self.samples):,}")

    @staticmethod
    def _load(data_path: str) -> list[dict]:
        if not data_path.endswith(".jsonl"):
            raise ValueError(f"AgentRLDataset 仅支持 .jsonl，收到 {data_path!r}")
        out: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def _parse_messages_and_tools(convs: list[dict]) -> tuple[list[dict], list[dict] | None]:
        """从 conversations 中分离出 messages 与 tools"""
        messages: list[dict] = []
        tools: list[dict] | None = None
        for raw in convs:
            msg = _normalize_message(raw)
            if msg.get("role") == "system" and msg.get("tools"):
                # tools 字段可能是 JSON 字符串或对象
                t = msg["tools"]
                tools = json.loads(t) if isinstance(t, str) else t
            if "tool_calls" in msg and isinstance(msg["tool_calls"], str):
                msg["tool_calls"] = json.loads(msg["tool_calls"])
            messages.append(msg)
        return messages, tools

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        messages, tools = self._parse_messages_and_tools(sample.get("conversations", []))
        # 通常去掉最后一条 assistant，让 rollout 重新生成
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return {
            "messages": messages,
            "tools": tools,
            "gt": sample.get("gt"),
        }

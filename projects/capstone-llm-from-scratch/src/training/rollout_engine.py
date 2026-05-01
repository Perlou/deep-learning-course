"""
rollout_engine.py — 可插拔 Rollout 引擎
=========================================

为 PPO / GRPO / Agentic RL 训练抽象一层"用 policy 模型生成 K 条候选 response"
的接口。不同后端（PyTorch ``model.generate`` vs SGLang HTTP）通过统一接口替换，
让 trainer 代码与生成后端解耦。

参考 minimind/trainer/rollout_engine.py 的设计：
  - ``RolloutEngine`` 抽象基类（``rollout()`` + ``update_policy()``）
  - ``TorchRolloutEngine``：直接用 model.generate()，CPU/MPS/单卡 GPU 都能跑
  - ``SGLangRolloutEngine``：HTTP POST 调用本地 sglang server，多卡时显著加速
  - ``RolloutResult`` 统一返回值（output_ids + completion_ids + per_token_logps + ...）

典型流程（GRPO/PPO）：

    engine = create_rollout_engine("torch", policy_model=actor, tokenizer=tk, device=device)
    result = engine.rollout(prompt_ids, attention_mask, num_generations=K, max_new_tokens=512)
    # result.completions 给 reward function 打分
    # result.per_token_logps 作为 old_logps 用于 PPO 比率计算
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# 工具函数
# ============================================================


def _per_token_logps(
    model: torch.nn.Module,
    input_ids: Tensor,
    n_keep: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """计算 ``input_ids`` 末尾 ``n_keep`` 个 token 的 log-prob

    用于 rollout 后给 trainer 提供"old policy log-prob"（PPO 比率分母）。

    Args:
        model:           当前策略模型（已 wrap 过 DDP / compile 也兼容）
        input_ids:       [B, T] 完整 token 序列（含 prompt + completion）
        n_keep:          需要的尾部 token 数（一般等于 max_new_tokens 或 completion_len）
        attention_mask:  [B, T] 0/1 mask（pad 位置为 0）

    Returns:
        [B, n_keep] 的 log-prob，对应 ``input_ids[:, -n_keep:]`` 每个 token 的概率
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # 取出真实 module（兼容 DDP / torch.compile）
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)

    out = unwrapped(input_ids, attention_mask=attention_mask)
    logits = out[0] if isinstance(out, tuple) else out
    # 用 logits[:, :-1, :] 预测 input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
    # 取末尾 n_keep（注意 shift 后总长 T-1）
    return token_logps[:, -n_keep:]


# ============================================================
# 通用返回值
# ============================================================


@dataclass
class RolloutResult:
    """统一的 rollout 返回结构

    Attributes:
        output_ids:       [B*K, P+R] 完整序列（prompt + completion）
        completion_ids:   [B*K, R]   仅 completion 部分（trim padding 后等长 padding）
        per_token_logps:  [B*K, R]   completion 每个 token 的 old log-prob
        completions:      list[str]  长度 B*K 的解码字符串
        prompt_lens:      [B*K]      每条样本的 prompt 真实长度
        completion_mask:  [B*K, R]   1=有效 completion token，0=pad
    """

    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: list[str]
    prompt_lens: Tensor
    completion_mask: Tensor


# ============================================================
# 抽象基类
# ============================================================


class RolloutEngine(ABC):
    """Rollout 引擎抽象基类"""

    tokenizer = None  # 子类必须设置

    @abstractmethod
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        """从 prompt 生成 num_generations 条 response"""

    @abstractmethod
    def update_policy(self, model: torch.nn.Module) -> None:
        """更新引擎内部持有的 policy 模型（trainer 每次 optimizer.step 后调）"""


# ============================================================
# 实现 1: PyTorch 原生（model.generate）
# ============================================================


class TorchRolloutEngine(RolloutEngine):
    """直接用 ``ClearMind GPT.generate`` 做 rollout

    优点：零依赖、与训练共享显存（适合单卡）
    缺点：generate 速度慢（无 KV cache 优化外的并发）；多卡时浪费

    适合：本地 / 单卡 GPU 调试 + 中小规模 GRPO 训练
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer,
        device: str | torch.device = "cuda",
        autocast_ctx=None,
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx

    def update_policy(self, model: torch.nn.Module) -> None:
        self.policy_model = model

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        """生成 num_generations 条 response

        实现策略：
          - 把 prompt repeat_interleave num_generations 次（[B, P] → [B*K, P]）
          - 调用 ``ClearMind generate()``（带 KV cache、top-k/top-p 采样）
          - 返回时拆出 prompt_lens / completion_mask
        """
        from src.inference.generate import generate

        # unwrap DDP / compile
        model = self.policy_model.module if hasattr(self.policy_model, "module") else self.policy_model
        model = getattr(model, "_orig_mod", model)

        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with torch.no_grad(), ctx:
            # repeat 到 [B*K, P]
            expanded_prompt = prompt_ids.repeat_interleave(num_generations, dim=0)
            expanded_mask = attention_mask.repeat_interleave(num_generations, dim=0)

            # generate() 一次只接受 batch，逐条 generate（因为 generate 内部对单 batch 优化）
            # 这里简化：循环每条，最后 stack
            all_outputs: list[Tensor] = []
            for i in range(expanded_prompt.size(0)):
                # generate 接受 [1, P]
                single_prompt = expanded_prompt[i : i + 1]
                output = generate(
                    model=model,
                    input_ids=single_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    eos_token_id=self.tokenizer.eos_id,
                )
                all_outputs.append(output[0])

            # pad 到统一长度（不同 generation 早停长度可能不同）
            max_len = max(o.size(0) for o in all_outputs)
            pad_id = self.tokenizer.pad_id
            padded = torch.full(
                (len(all_outputs), max_len),
                pad_id,
                dtype=expanded_prompt.dtype,
                device=expanded_prompt.device,
            )
            for i, o in enumerate(all_outputs):
                padded[i, : o.size(0)] = o
            output_ids = padded

            # 切出 completion 部分
            prompt_len = expanded_prompt.size(1)
            completion_ids = output_ids[:, prompt_len:]
            full_mask = (output_ids != pad_id).long()

            # 计算 per-token logps（用当前 policy）
            per_token_logps = _per_token_logps(
                self.policy_model,
                output_ids,
                completion_ids.size(1),
                attention_mask=full_mask,
            )

        completions = [
            self.tokenizer.decode(c.tolist(), skip_special_tokens=True)
            for c in completion_ids
        ]
        prompt_lens = expanded_prompt.new_full((output_ids.size(0),), prompt_len)
        # completion_mask: completion 部分非 pad 即 1
        completion_mask = (completion_ids != pad_id).long()

        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_lens,
            completion_mask=completion_mask,
        )


# ============================================================
# 实现 2: SGLang HTTP（生产 RL 训练时强烈推荐）
# ============================================================


class SGLangRolloutEngine(RolloutEngine):
    """通过 HTTP 调用本地 SGLang server 做 rollout

    优点：sglang 用 RadixAttention 共享 prompt KV cache，推理 5-20× 快
    缺点：需要先 ``python -m sglang.launch_server --model-path ./model_dir``
    用法：先把当前 policy 权重 ``save_pretrained`` 到 ``shared_ckpt_path``，
          然后 ``http POST /update_weights_from_disk`` 让 sglang reload

    参考 minimind/trainer/rollout_engine.py 的实现。仅提供占位骨架，
    生产使用前需要适配 ClearMind 模型 → Qwen3 兼容权重的转换流程（Phase 5）。
    """

    def __init__(
        self,
        base_url: str,
        model_path: str,
        shared_ckpt_path: str = "./sglang_ckpt",
        timeout: int = 120,
    ):
        try:
            import requests
        except ImportError as e:
            raise ImportError("SGLangRolloutEngine 需要 `pip install requests`") from e
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._http = requests

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        # 去除左侧 padding
        input_ids_list: list[list[int]] = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid = ids[mask.bool()].tolist()
            input_ids_list.append(valid)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": (
                    [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
                ),
            },
            "return_logprob": True,
        }
        resp = self._http.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        completions: list[str] = []

        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            comp = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            logps = [
                (item[0] if isinstance(item, (list, tuple)) and len(item) >= 1 else item)
                for item in raw_logprobs
            ]
            if len(logps) < len(comp):
                logps = [0.0] * (len(comp) - len(logps)) + logps
            elif len(logps) > len(comp):
                logps = logps[-len(comp):] if comp else []

            prompt = all_input_ids[i]
            all_output_ids.append(prompt + comp)
            all_completion_ids.append(comp)
            all_logprobs.append(logps)
            completions.append(self.tokenizer.decode(comp, skip_special_tokens=True))

        device = prompt_ids.device
        max_comp = max(1, max(len(ids) for ids in all_completion_ids))
        max_out = max(len(ids) for ids in all_input_ids) + max_comp

        def pad(seqs: list[list], max_len: int, pad_val):
            return torch.tensor(
                [s + [pad_val] * (max_len - len(s)) for s in seqs], device=device
            )

        pad_id = self.tokenizer.pad_token_id or 0
        return RolloutResult(
            output_ids=pad(all_output_ids, max_out, pad_id),
            completion_ids=pad(all_completion_ids, max_comp, pad_id),
            per_token_logps=pad(all_logprobs, max_comp, 0.0).float(),
            completions=completions,
            prompt_lens=torch.tensor(
                [len(ids) for ids in all_input_ids], device=device
            ),
            completion_mask=torch.tensor(
                [
                    [1] * len(ids) + [0] * (max_comp - len(ids))
                    for ids in all_completion_ids
                ],
                device=device,
            ),
        )

    def update_policy(self, model: torch.nn.Module) -> None:
        """把当前 policy 权重 dump 到 ``shared_ckpt_path`` 并通知 sglang reload"""
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
        abs_path = os.path.abspath(self.shared_ckpt_path)
        # save_pretrained 需要模型继承自 PreTrainedModel；ClearMind GPT 当前还没继承
        # 这里走通用路径：保存为 .pth + tokenizer 配置（sglang 端可加载）
        if hasattr(unwrapped, "save_pretrained"):
            sd = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
            unwrapped.save_pretrained(abs_path, state_dict=sd, safe_serialization=False)
        else:
            os.makedirs(abs_path, exist_ok=True)
            torch.save(
                {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()},
                os.path.join(abs_path, "pytorch_model.bin"),
            )
        self.tokenizer.save_pretrained(abs_path)

        resp = self._http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"sglang update_weights_from_disk 失败: {resp.text}")


# ============================================================
# 工厂
# ============================================================


def create_rollout_engine(
    engine_type: str = "torch",
    *,
    policy_model: torch.nn.Module = None,
    tokenizer=None,
    device: str | torch.device = "cuda",
    autocast_ctx=None,
    sglang_base_url: str | None = None,
    sglang_model_path: str | None = None,
    sglang_shared_path: str | None = None,
) -> RolloutEngine:
    """工厂：根据 engine_type 返回对应实现"""
    if engine_type == "torch":
        if policy_model is None or tokenizer is None:
            raise ValueError("TorchRolloutEngine 需要 policy_model + tokenizer")
        return TorchRolloutEngine(
            policy_model=policy_model,
            tokenizer=tokenizer,
            device=device,
            autocast_ctx=autocast_ctx,
        )
    if engine_type == "sglang":
        if not sglang_base_url or not sglang_model_path:
            raise ValueError("SGLangRolloutEngine 需要 sglang_base_url + sglang_model_path")
        return SGLangRolloutEngine(
            base_url=sglang_base_url,
            model_path=sglang_model_path,
            shared_ckpt_path=sglang_shared_path or "./sglang_ckpt",
        )
    raise ValueError(f"未知 engine_type: {engine_type!r}（应为 'torch' / 'sglang'）")

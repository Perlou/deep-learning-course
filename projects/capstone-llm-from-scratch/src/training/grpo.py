"""
grpo.py — GRPO + CISPO Trainer
================================

DeepSeek-R1 同款的 RL 训练算法：

  GRPO（Group Relative Policy Optimization）
    每个 prompt 采样 K 条 response，用组内 ``(reward - mean) / std`` 当 advantage，
    避免训练独立的 critic（PPO 需要）。
    Loss = -E[ min(ratio·A, clip(ratio, 1±ε)·A) - β·KL_ref ]

  CISPO（GRPO 的稳定性变体）
    用 ``clamped_ratio = clamp(ratio, max=ε_high).detach()`` 取代双向 clip，
    Loss = -(clamped_ratio · A · log_p - β·KL_ref)
    工程上更稳定，适合长序列高方差场景。

参考：minimind/trainer/train_grpo.py、DeepSeek-R1 paper

Reward 来源：
  - 简单规则奖励（长度 / 格式 / 重复惩罚），不需要外部 reward model
  - 可选外接 HF reward model（通过 LMForRewardModel 适配器，待 P3-5）

典型用法：
  python scripts/train.py --stage grpo \\
      --config configs/main.yaml \\
      --resume outputs/sft/final.pth \\
      --data data/rlaif.jsonl
"""

from __future__ import annotations

import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .trainer_utils import (
    CosineWarmupScheduler,
    load_checkpoint,
    amp_autocast,
)
from .rollout_engine import create_rollout_engine, _per_token_logps


# ============================================================
# 简单规则 Reward（无需外部 reward model 即可启动 GRPO）
# ============================================================


def _ngram_repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """统计 n-gram 重复率，最大 ``cap``，越高越差（用作惩罚）"""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / max(len(grams), 1))


def default_rule_reward(prompts: list[str], responses: list[str]) -> torch.Tensor:
    """简单规则 reward（适合训练初期 / 没有 reward model 时的 baseline）

    评分规则：
      - 长度 [20, 800] 字符 → +0.5，否则 -0.5
      - 含 ``</think>`` 且 think 内容 [20, 300] → +1.0；否则 -0.5
      - 仅出现一次 ``</think>`` → +0.25，多次 → -0.25
      - 减去 trigram 重复率（cap=0.5）

    Returns:
        ``[len(responses)]`` reward 张量
    """
    rewards = torch.zeros(len(responses))
    for i, resp in enumerate(responses):
        r = 0.0
        # 长度
        if 20 <= len(resp.strip()) <= 800:
            r += 0.5
        else:
            r -= 0.5

        # think 标签
        answer = resp
        if "</think>" in resp:
            think, _, ans = resp.partition("</think>")
            think = think.replace("<think>", "").strip()
            ans = ans.strip()
            r += 1.0 if 20 <= len(think) <= 300 else -0.5
            r += 0.25 if resp.count("</think>") == 1 else -0.25
            answer = ans

        # 重复惩罚
        r -= _ngram_repetition_penalty(answer)
        rewards[i] = r
    return rewards


# ============================================================
# GRPO Trainer
# ============================================================


class GRPOTrainer(BaseTrainer):
    """GRPO + CISPO RL Trainer

    Args:
        model:         policy 模型（要训练的）
        train_dataset: :class:`RLAIFDataset` 风格（每条样本含 prompt 字符串）
        config:        训练配置
        val_dataset:   验证集（可选）
        output_dir:    输出目录
        reward_fn:     reward 函数 ``(prompts, completions) -> Tensor``
                        默认用 :func:`default_rule_reward`

    Config 关键字段：
      - ``epochs``、``batch_size``、``lr`` 等同 BaseTrainer
      - ``num_generations``: 每个 prompt 采样的 response 数（默认 6）
      - ``loss_type``: ``"grpo"`` 或 ``"cispo"``（默认 cispo，更稳）
      - ``epsilon``: GRPO clip 范围（默认 0.2）
      - ``epsilon_high``: CISPO 单边 clip 上界（默认 5.0）
      - ``beta``: KL ref penalty 系数（默认 0.1）
      - ``max_gen_len``: 单条 response 最长 token 数（默认 1024）
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
        output_dir: str = "outputs/grpo",
        reward_fn=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            config=config,
            val_dataset=val_dataset,
            output_dir=output_dir,
            stage_name="GRPO 训练",
            default_lr=3e-7,
            default_batch_size=2,
            default_grad_accum=1,
            default_patience=3,
            default_log_every=1,
            early_stopping_mode="max",  # reward 越大越好
        )

        self.num_generations = int(config.get("num_generations", 6))
        self.loss_type = config.get("loss_type", "cispo")
        self.epsilon = float(config.get("epsilon", 0.2))
        self.epsilon_high = float(config.get("epsilon_high", 5.0))
        self.beta = float(config.get("beta", 0.1))
        self.max_gen_len = int(config.get("max_gen_len", 1024))
        self.pad_token_id = int(config.get("pad_token_id", 0))
        self.reward_fn = reward_fn or default_rule_reward

        self.epochs = int(config.get("epochs", 1))
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation)
        )
        self.max_steps = self.steps_per_epoch * self.epochs

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            max_lr=self.lr,
            min_lr=self.lr * 0.1,
            warmup_steps=min(50, self.max_steps // 10),
            max_steps=self.max_steps,
        )

        self.ref_model: nn.Module | None = None
        self.rollout_engine = None

        print("\n📋 GRPO 配置:")
        print(f"  Loss type:        {self.loss_type}")
        print(f"  num_generations:  {self.num_generations}")
        print(f"  epsilon (clip):   {self.epsilon}")
        if self.loss_type == "cispo":
            print(f"  epsilon_high:     {self.epsilon_high}")
        print(f"  beta (KL):        {self.beta}")
        print(f"  max_gen_len:      {self.max_gen_len}")
        print(f"  Total steps:      {self.max_steps}")
        print(f"  LR:               {self.lr}")

    def _grpo_collate(self, batch: list[dict]) -> dict:
        """GRPO 用 RLAIFDataset，每条返回 dict(prompt, answer, messages)"""
        return {
            "prompts": [b["prompt"] for b in batch],
            "messages": [b.get("messages") for b in batch],
            "answers": [b.get("answer", "") for b in batch],
        }

    def train(self, sft_path: str | None = None) -> None:
        """执行 GRPO 训练

        Args:
            sft_path: 可选 SFT checkpoint（policy 起点）；不传则用当前 self.model 权重
        """
        # ---- 0. 自动续训 / 加载 SFT 权重 ----
        auto_info = self._try_auto_resume()
        start_step = 0
        if auto_info is not None:
            start_step = auto_info.get("step", 0)
            print(f"🔄 自动从 step={start_step} 续训 GRPO")
        elif sft_path and os.path.exists(sft_path):
            load_checkpoint(self.model, sft_path, device=self.device)
            print(f"✅ Policy 已加载 SFT 权重: {sft_path}")

        # ---- 1. 创建 reference model（冻结 SFT 模型，用于 KL ref）----
        from ..model.gpt import GPT

        print("📋 创建 reference 模型 ...")
        self.ref_model = GPT(self.model.config).to(self.device)
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # ---- 2. 创建 rollout 引擎（默认 torch）----
        self.rollout_engine = create_rollout_engine(
            engine_type=self.config.get("rollout_engine", "torch"),
            policy_model=self.model,
            tokenizer=getattr(self.train_loader.dataset, "tokenizer", None),
            device=self.device,
        )

        # ---- 3. 替换 DataLoader 的 collate（GRPO 输入是 prompt 字符串）----
        from torch.utils.data import DataLoader

        rl_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._grpo_collate,
            num_workers=0,
        )

        print(f"\n{'=' * 60}")
        print(f"🚀 开始 GRPO ({self.epochs} epochs, {self.max_steps} steps)")
        print(f"{'=' * 60}\n")

        self.model.train()
        step = start_step
        tokenizer = self.rollout_engine.tokenizer
        pad_id = self.pad_token_id

        for epoch in range(self.epochs):
            print(f"\n📅 Epoch {epoch + 1}/{self.epochs}")
            for batch in rl_loader:
                if step >= self.max_steps:
                    break
                prompts: list[str] = batch["prompts"]

                # ---- 3.1 tokenize prompts ----
                enc = [tokenizer.encode(p, add_bos=False, add_eos=False) for p in prompts]
                max_p_len = max(len(e) for e in enc)
                # 左 pad（适合 GRPO/PPO 生成）
                prompt_ids = torch.full(
                    (len(enc), max_p_len), pad_id, dtype=torch.long, device=self.device
                )
                attn_mask = torch.zeros_like(prompt_ids)
                for i, e in enumerate(enc):
                    prompt_ids[i, max_p_len - len(e) :] = torch.tensor(e, device=self.device)
                    attn_mask[i, max_p_len - len(e) :] = 1

                # ---- 3.2 rollout ----
                with torch.no_grad():
                    rollout = self.rollout_engine.rollout(
                        prompt_ids=prompt_ids,
                        attention_mask=attn_mask,
                        num_generations=self.num_generations,
                        max_new_tokens=self.max_gen_len,
                        temperature=0.8,
                    )

                B = len(prompts)
                K = self.num_generations
                completions = rollout.completions  # 长度 B*K
                old_logps = rollout.per_token_logps.to(self.device)  # [B*K, R]
                completion_mask = rollout.completion_mask.to(self.device).float()  # [B*K, R]
                output_ids = rollout.output_ids.to(self.device)
                full_mask = (output_ids != pad_id).long()
                # logp_pos：用于在完整序列 logits 中提取 completion 的位置
                completion_len = rollout.completion_ids.size(1)

                # ---- 3.3 计算 reward + group-relative advantage ----
                # 把每个 prompt 重复 K 次以匹配 K 条 response
                expanded_prompts = [p for p in prompts for _ in range(K)]
                rewards = self.reward_fn(expanded_prompts, completions).to(self.device)
                # group-relative
                grouped = rewards.view(B, K)
                mean_r = grouped.mean(dim=1).repeat_interleave(K)
                std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(K).clamp(min=1e-4)
                advantages = (rewards - mean_r) / std_r  # [B*K]

                # ---- 3.4 当前 policy + reference policy logps ----
                with amp_autocast(self.device, self.dtype):
                    cur_logps = _per_token_logps(
                        self.model, output_ids, completion_len, attention_mask=full_mask
                    )
                with torch.no_grad():
                    ref_logps = _per_token_logps(
                        self.ref_model, output_ids, completion_len, attention_mask=full_mask
                    )

                # ---- 3.5 KL 项（k3 estimator: e^kl - kl - 1，无偏低方差）----
                kl_div = ref_logps - cur_logps
                per_token_kl = torch.exp(kl_div) - kl_div - 1.0  # >= 0

                # ---- 3.6 比率 ratio = exp(cur - old) ----
                ratio = torch.exp(cur_logps - old_logps)
                adv = advantages.unsqueeze(1)  # [B*K, 1]

                if self.loss_type == "cispo":
                    # CISPO: clamp(ratio, max=ε_high).detach() · adv · log_p - β·kl
                    clamped = torch.clamp(ratio, max=self.epsilon_high).detach()
                    per_token_loss = -(clamped * adv * cur_logps - self.beta * per_token_kl)
                else:
                    # GRPO 标准 PPO-clip
                    clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    per_token_loss = -(
                        torch.min(ratio * adv, clipped * adv) - self.beta * per_token_kl
                    )

                # 仅对有效 completion token 取平均
                policy_loss = (
                    (per_token_loss * completion_mask).sum(dim=1)
                    / completion_mask.sum(dim=1).clamp(min=1)
                ).mean()

                loss = policy_loss / self.gradient_accumulation
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                grad_norm, lr = self._optimizer_step()

                # ---- 4. 日志 ----
                avg_reward = rewards.mean().item()
                kl_ref = ((ref_logps - cur_logps) * completion_mask).sum().item() / max(
                    completion_mask.sum().item(), 1
                )
                avg_len = completion_mask.sum(dim=1).float().mean().item()

                self.logger.log(
                    step=step,
                    max_steps=self.max_steps,
                    loss=policy_loss.item(),
                    lr=lr,
                    grad_norm=grad_norm,
                )
                if self.logger.tb_writer:
                    self.logger.tb_writer.add_scalar("grpo/reward", avg_reward, step)
                    self.logger.tb_writer.add_scalar("grpo/kl_ref", kl_ref, step)
                    self.logger.tb_writer.add_scalar("grpo/avg_response_len", avg_len, step)

                if step % 5 == 0:
                    print(
                        f"  step={step}  reward={avg_reward:.4f}  "
                        f"kl_ref={kl_ref:.4f}  avg_len={avg_len:.0f}  "
                        f"adv_std={advantages.std().item():.4f}"
                    )

                # ---- 5. checkpoint + rollout engine 同步 ----
                if (step + 1) % self.save_every == 0:
                    self._save(
                        step + 1, policy_loss.item(), f"checkpoint_step{step + 1}.pth"
                    )
                    self.rollout_engine.update_policy(self.model)

                step += 1

            if step >= self.max_steps:
                break

        self._finalize(step, policy_loss.item(), "grpo_log.jsonl")
        del self.ref_model
        self.ref_model = None

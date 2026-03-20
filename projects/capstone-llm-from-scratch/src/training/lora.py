"""
lora.py — LoRA 低秩适配微调
============================

实现 Low-Rank Adaptation (LoRA), 用于高效微调大语言模型。

LoRA 核心思想:
  冻结预训练权重 W, 只训练低秩分解 ΔW = B @ A
  其中 A ∈ R^(r × d_in), B ∈ R^(d_out × r), r << min(d_in, d_out)

优势:
  - 显存占用大幅降低 (只训练 LoRA 参数)
  - 训练速度更快
  - 可以保存/切换多个 LoRA 适配器
  - 部署时可合并回原始权重, 不增加推理开销

典型用法:
  apply_lora(model, rank=8, target_modules=["w_q", "w_v"])
  # ... 训练 ...
  merge_lora(model)  # 合并回原始权重
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA 适配的线性层

    在原始 Linear 层旁边添加低秩分支:
      output = W @ x + (B @ A) @ x * (alpha / rank)

    Args:
        original_linear: 原始的 nn.Linear 层
        rank:            LoRA 秩 (r), 控制参数量和表达能力
        alpha:           LoRA 缩放因子, 通常 alpha=rank 或 alpha=2*rank
        dropout:         LoRA dropout, 防止过拟合
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 保留原始权重 (冻结)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA 低秩矩阵
        # A: 降维 (d_in → r), Kaiming 初始化
        # B: 升维 (r → d_out), 零初始化 → 初始时 ΔW = 0
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # 初始化 A (Kaiming uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 原始输出 + LoRA 低秩修正"""
        # 原始 Linear 输出
        result = F.linear(x, self.weight, self.bias)

        # LoRA 分支: x → dropout → A → B → scale
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T  # [*, r]
        lora_out = lora_out @ self.lora_B.T  # [*, d_out]
        result = result + lora_out * self.scaling

        return result

    def merge(self):
        """将 LoRA 权重合并回原始权重

        合并后模型等价于原始结构, 推理无额外开销。
        """
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling

    @property
    def lora_params(self) -> int:
        """LoRA 参数量"""
        return self.rank * (self.in_features + self.out_features)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] = None,
) -> nn.Module:
    """在模型上应用 LoRA

    冻结原始参数, 只保留 LoRA 参数可训练。

    Args:
        model:          原始模型
        rank:           LoRA 秩
        alpha:          LoRA 缩放因子
        dropout:        LoRA dropout
        target_modules: 要适配的模块名 (默认: w_q, w_k, w_v, w_o)

    Returns:
        应用了 LoRA 的模型 (原地修改)
    """
    if target_modules is None:
        target_modules = ["w_q", "w_k", "w_v", "w_o"]

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换目标 Linear 层为 LoRALinear
    replaced = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if not hasattr(module, target):
                continue

            original = getattr(module, target)
            if not isinstance(original, nn.Linear):
                continue

            lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, target, lora_layer)
            replaced += 1

    # LoRA 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🔧 LoRA 已应用:")
    print(f"  替换层数:    {replaced}")
    print(f"  Rank:        {rank}")
    print(f"  Alpha:       {alpha}")
    print(f"  全部参数:    {total_params:,}")
    print(
        f"  可训练参数:  {trainable_params:,} ({trainable_params / total_params:.2%})"
    )

    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """合并所有 LoRA 权重回原始模型

    合并后模型等价于原始 Linear 结构, 推理无额外开销。

    Returns:
        合并后的模型 (原地修改)
    """
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged += 1

    print(f"🔗 LoRA 合并完成: {merged} 层")
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """只提取 LoRA 参数的 state_dict

    用于保存/加载 LoRA 适配器, 体积远小于完整模型。

    Returns:
        只包含 LoRA 参数的字典
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params[name] = param.data
    return lora_params


def load_lora_state_dict(model: nn.Module, state_dict: dict):
    """加载 LoRA 参数

    Args:
        model:      已应用 LoRA 的模型
        state_dict: LoRA 参数字典
    """
    model_dict = model.state_dict()
    lora_keys = [k for k in state_dict if "lora_A" in k or "lora_B" in k]

    for key in lora_keys:
        if key in model_dict:
            model_dict[key] = state_dict[key]

    model.load_state_dict(model_dict, strict=False)
    print(f"📦 LoRA 参数加载: {len(lora_keys)} 个张量")

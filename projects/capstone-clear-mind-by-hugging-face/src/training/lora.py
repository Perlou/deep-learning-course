"""
lora.py — LoRA / QLoRA 低秩适配微调 (PEFT 版)
==============================================

使用 HuggingFace PEFT 库实现 LoRA 和 QLoRA，替代 from-scratch 的手写 LoRALinear。

from-scratch 对比:
  - LoRALinear(nn.Module) 手写低秩分支 → PEFT get_peft_model() 自动注入
  - apply_lora() 手动替换 Linear → get_peft_model() 一行完成
  - merge_lora() 手动合并权重 → model.merge_and_unload() 一行完成
  - lora_state_dict() 手动提取 → model.save_pretrained() 自动保存 adapter
  - load_lora_state_dict() 手动加载 → PeftModel.from_pretrained() 自动加载
  - (无 QLoRA) → BitsAndBytesConfig + LoRA 实现 4-bit 量化微调

LoRA 核心原理 (两种实现完全等价):
  W' = W + B @ A * (alpha / r)
  其中 A ∈ R^(r × d_in), B ∈ R^(d_out × r)

QLoRA 核心原理 (from-scratch 版没有):
  1. 将基础模型量化为 4-bit (NF4 数据类型)
  2. 在量化模型上应用 LoRA
  3. 训练时只更新 LoRA 参数 (FP16/BF16 精度)
  4. 显存占用 ≈ 全精度的 1/4，训练效果接近全参微调
"""

import yaml
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training


def create_lora_config(lora_config: dict) -> LoraConfig:
    """从 YAML lora 节创建 LoraConfig

    tiny.yaml lora 节:
      r: 8, lora_alpha: 16, lora_dropout: 0.05,
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

    from-scratch 对比:
      from-scratch 直接传 rank, alpha, dropout, target_modules 参数
      PEFT 用 LoraConfig dataclass 封装

    Args:
        lora_config: YAML 中 lora 节的字典

    Returns:
        LoraConfig 实例
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        bias="none",
    )


def apply_peft_lora(model, lora_config: LoraConfig):
    """在模型上应用 LoRA

    from-scratch 对比:
      - from-scratch: apply_lora(model, rank, alpha, ...) 手动替换每个 Linear
      - PEFT: get_peft_model(model, config) 一行完成

    Args:
        model: ClearMindForCausalLM 实例
        lora_config: LoraConfig 实例

    Returns:
        PeftModel (包裹了 LoRA adapter 的模型)
    """
    peft_model = get_peft_model(model, lora_config)

    # 打印参数统计 (对应 from-scratch apply_lora 的打印)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"\nLoRA 已应用 (PEFT):")
    print(f"  Rank:        {lora_config.r}")
    print(f"  Alpha:       {lora_config.lora_alpha}")
    print(f"  Target:      {lora_config.target_modules}")
    print(f"  全部参数:    {total:,}")
    print(f"  可训练参数:  {trainable:,} ({trainable / total:.2%})")

    return peft_model


def merge_lora_weights(model):
    """合并 LoRA 权重回原始模型

    from-scratch 对比:
      - from-scratch: merge_lora(model) 手动 W += B @ A * scaling
      - PEFT: model.merge_and_unload() 一行完成

    Args:
        model: PeftModel 实例

    Returns:
        合并后的原始模型 (不再是 PeftModel)
    """
    merged = model.merge_and_unload()
    print("LoRA 权重已合并 (merge_and_unload)")
    return merged


def create_qlora_config(lora_config: dict) -> tuple:
    """创建 QLoRA 配置 (BitsAndBytesConfig + LoraConfig)

    QLoRA = 4-bit 量化 + LoRA，显存占用约为全精度的 1/4。
    from-scratch 版没有 QLoRA 支持。

    需要:
      - CUDA GPU (bitsandbytes 不支持 CPU/MPS)
      - pip install bitsandbytes

    Args:
        lora_config: YAML 中 lora 节的字典

    Returns:
        (BitsAndBytesConfig, LoraConfig) 元组

    Raises:
        ImportError: 如果 bitsandbytes 未安装或不在 CUDA 环境
    """
    try:
        from transformers import BitsAndBytesConfig
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA 需要 CUDA GPU。当前环境不支持 CUDA。"
                "请在 GPU 服务器上运行，或使用普通 LoRA (--use_lora)。"
            )

        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NormalFloat4 量化
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
            bnb_4bit_use_double_quant=True,       # 二次量化，进一步节省显存
        )

        # LoRA 配置 (与普通 LoRA 相同)
        lora_cfg = create_lora_config(lora_config)

        return bnb_config, lora_cfg

    except ImportError as e:
        raise ImportError(
            "QLoRA 需要 bitsandbytes 库。请安装: pip install bitsandbytes\n"
            f"原始错误: {e}"
        ) from e


def load_model_qlora(model_path: str, bnb_config, lora_config: LoraConfig):
    """加载量化模型并应用 LoRA (QLoRA 完整流程)

    QLoRA 流程:
      1. 用 BitsAndBytesConfig 加载 4-bit 量化模型
      2. prepare_model_for_kbit_training 准备量化模型
      3. get_peft_model 应用 LoRA

    Args:
        model_path:  模型路径 (HF 格式目录)
        bnb_config:  BitsAndBytesConfig (4-bit 量化配置)
        lora_config: LoraConfig

    Returns:
        PeftModel (量化 + LoRA 的模型)
    """
    from model import ClearMindForCausalLM

    # 1. 加载 4-bit 量化模型
    model = ClearMindForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 2. 准备量化模型用于训练
    model = prepare_model_for_kbit_training(model)

    # 3. 应用 LoRA
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"\nQLoRA 已应用:")
    print(f"  量化: 4-bit NF4")
    print(f"  Rank: {lora_config.r}")
    print(f"  全部参数:    {total:,}")
    print(f"  可训练参数:  {trainable:,} ({trainable / total:.2%})")

    return peft_model

"""
配置管理
========

包含模型、LoRA、训练和数据的配置参数。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型配置"""

    vocab_size: int = 32000  # 词表大小
    hidden_size: int = 512  # 隐藏层维度
    intermediate_size: int = 2048  # FFN中间层维度
    num_layers: int = 6  # Transformer层数
    num_heads: int = 8  # 注意力头数
    max_seq_len: int = 512  # 最大序列长度
    dropout: float = 0.1  # Dropout率

    # 特殊token ID
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class LoRAConfig:
    """LoRA配置

    LoRA (Low-Rank Adaptation) 通过低秩分解减少微调参数量：
    ΔW = B × A，其中 A: (d, r), B: (r, d)
    """

    rank: int = 8  # 低秩分解的秩
    alpha: float = 16.0  # 缩放因子
    dropout: float = 0.05  # LoRA dropout
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]  # 应用LoRA的目标层
    )

    @property
    def scaling(self) -> float:
        """计算LoRA缩放因子"""
        return self.alpha / self.rank


@dataclass
class TrainingConfig:
    """训练配置"""

    # 基本训练参数
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3

    # 学习率调度
    warmup_ratio: float = 0.1  # warmup比例
    lr_scheduler_type: str = "cosine"

    # 保存和日志
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50

    # 其他
    max_grad_norm: float = 1.0  # 梯度裁剪
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""

    max_length: int = 256  # 最大token长度
    train_test_split: float = 0.1  # 测试集比例

    # ChatML模板标记
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    # 角色标记
    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = "assistant"

    # 默认系统提示
    default_system_prompt: str = "你是一个有帮助的AI助手。"


@dataclass
class Config:
    """总配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 输出路径
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    log_dir: str = "outputs/logs"


def get_config() -> Config:
    """获取默认配置"""
    return Config()


def print_config(config: Config):
    """打印配置信息"""
    print("=" * 60)
    print("配置信息")
    print("=" * 60)

    print(f"\n模型配置:")
    print(f"  词表大小: {config.model.vocab_size}")
    print(f"  隐藏维度: {config.model.hidden_size}")
    print(f"  层数: {config.model.num_layers}")
    print(f"  注意力头数: {config.model.num_heads}")

    print(f"\nLoRA配置:")
    print(f"  秩 (rank): {config.lora.rank}")
    print(f"  alpha: {config.lora.alpha}")
    print(f"  目标层: {config.lora.target_modules}")

    print(f"\n训练配置:")
    print(f"  批次大小: {config.training.batch_size}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  训练轮数: {config.training.num_epochs}")


if __name__ == "__main__":
    config = get_config()
    print_config(config)

from .pretrain_dataset import PretrainDataset
from .sft_dataset import SFTDataset
from .dpo_dataset import DPODataset
from .rl_dataset import RLAIFDataset, AgentRLDataset

__all__ = [
    "ClearMindTokenizer",
    "HFTokenizer",
    "PretrainDataset",
    "SFTDataset",
    "DPODataset",
    "RLAIFDataset",
    "AgentRLDataset",
]


def __getattr__(name):
    """Lazy import 两类 tokenizer，避免在仅使用 dataset 时强制安装 sentencepiece / transformers。"""
    if name == "ClearMindTokenizer":
        from .tokenizer import ClearMindTokenizer

        return ClearMindTokenizer
    if name == "HFTokenizer":
        from .hf_tokenizer import HFTokenizer

        return HFTokenizer
    raise AttributeError(f"module 'data' has no attribute {name!r}")

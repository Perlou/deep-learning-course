from .pretrain_dataset import PretrainDataset
from .sft_dataset import SFTDataset
from .dpo_dataset import DPODataset

__all__ = ["ClearMindTokenizer", "PretrainDataset", "SFTDataset", "DPODataset"]


def __getattr__(name):
    """Lazy import tokenizer to avoid forcing sentencepiece for dataset-only usage."""
    if name == "ClearMindTokenizer":
        from .tokenizer import ClearMindTokenizer

        return ClearMindTokenizer
    raise AttributeError(f"module 'data' has no attribute '{name}'")

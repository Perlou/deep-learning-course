"""
export_gguf.py — GGUF 格式导出 (HF 版)
=========================================

将 ClearMind-HF 模型导出为 GGUF 格式，支持 llama.cpp 推理。

from-scratch 对比:
  - from-scratch: GPT + load_checkpoint + w_q/w_k/w_v 权重名映射
  - HF 版: from_pretrained + q_proj/k_proj/v_proj 权重名映射

权重命名映射 (ClearMind-HF → llama.cpp):
  model.embed_tokens.weight         → token_embd.weight
  model.layers.{i}.self_attn.q_proj → blk.{i}.attn_q.weight
  model.layers.{i}.self_attn.k_proj → blk.{i}.attn_k.weight
  model.layers.{i}.self_attn.v_proj → blk.{i}.attn_v.weight
  model.layers.{i}.self_attn.o_proj → blk.{i}.attn_output.weight
  model.layers.{i}.input_layernorm  → blk.{i}.attn_norm.weight
  model.layers.{i}.mlp.gate_proj    → blk.{i}.ffn_gate.weight
  model.layers.{i}.mlp.up_proj      → blk.{i}.ffn_up.weight
  model.layers.{i}.mlp.down_proj    → blk.{i}.ffn_down.weight
  model.layers.{i}.post_attn_layernorm → blk.{i}.ffn_norm.weight
  model.norm.weight                 → output_norm.weight
  lm_head.weight                    → output.weight

用法:
  python deploy/export_gguf.py --model outputs/sft --output model-f16.gguf
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from model import ClearMindForCausalLM


# ============================================================
# GGUF 常量 (spec v3)
# ============================================================

GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


# ============================================================
# GGUF 写入器
# ============================================================


class GGUFWriter:
    """GGUF 文件写入器 (spec v3)"""

    ALIGNMENT = 32

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata: list[tuple] = []
        self.tensors: list[tuple] = []

    def add_string(self, key: str, value: str):
        self.metadata.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_UINT32, value))

    def add_float32(self, key: str, value: float):
        self.metadata.append((key, GGUF_TYPE_FLOAT32, value))

    def add_string_array(self, key: str, values: list[str]):
        self.metadata.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, values)))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = GGML_TYPE_F32):
        self.tensors.append((name, data, ggml_type))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, vtype: int, value):
        if vtype == GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type, values = value
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(values)))
            for v in values:
                self._write_metadata_value(f, elem_type, v)

    def _pad_to_alignment(self, f):
        pos = f.tell()
        pad = (self.ALIGNMENT - pos % self.ALIGNMENT) % self.ALIGNMENT
        if pad > 0:
            f.write(b"\x00" * pad)

    def write(self):
        with open(self.output_path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Metadata
            for key, vtype, value in self.metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                self._write_metadata_value(f, vtype, value)

            # Tensor info (first pass: placeholder offsets)
            tensor_info_start = f.tell()
            for name, data, ggml_type in self.tensors:
                self._write_string(f, name)
                f.write(struct.pack("<I", len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", 0))

            self._pad_to_alignment(f)
            data_start = f.tell()

            # Rewrite with correct offsets
            f.seek(tensor_info_start)
            current_offset = 0
            for name, data, ggml_type in self.tensors:
                self._write_string(f, name)
                f.write(struct.pack("<I", len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", current_offset))
                size = data.nbytes
                current_offset += size + (self.ALIGNMENT - size % self.ALIGNMENT) % self.ALIGNMENT

            # Tensor data
            f.seek(data_start)
            for _, data, _ in self.tensors:
                f.write(data.tobytes())
                pad = (self.ALIGNMENT - data.nbytes % self.ALIGNMENT) % self.ALIGNMENT
                if pad > 0:
                    f.write(b"\x00" * pad)

        file_size = os.path.getsize(self.output_path)
        print(f"GGUF 已写入: {self.output_path}")
        print(f"  大小: {file_size / 1024 / 1024:.1f} MB, 张量数: {len(self.tensors)}")


# ============================================================
# HF 权重名称映射
# ============================================================


def map_weight_name(hf_name: str) -> str | None:
    """ClearMind-HF → llama.cpp 权重命名映射

    from-scratch 对比:
      - from-scratch: w_q/w_k/w_v/w_o, w_gate/w_up/w_down
      - HF 版: q_proj/k_proj/v_proj/o_proj, gate_proj/up_proj/down_proj
    """
    mapping = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    if hf_name in mapping:
        return mapping[hf_name]

    # model.layers.{i}.xxx → blk.{i}.xxx
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])

        layer_mapping = {
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "input_layernorm.weight": "attn_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }

        if rest in layer_mapping:
            return f"blk.{layer_idx}.{layer_mapping[rest]}"

    return None


def export_gguf(
    model,
    output_path: str,
    dtype: str = "f16",
):
    """将 HF 模型导出为 GGUF 格式

    Args:
        model:       ClearMindForCausalLM
        output_path: 输出文件路径
        dtype:       导出精度 ("f32" 或 "f16")
    """
    config = model.config
    writer = GGUFWriter(output_path)

    # 元数据
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "ClearMind-HF")
    writer.add_uint32("llama.context_length", config.max_position_embeddings)
    writer.add_uint32("llama.embedding_length", config.hidden_size)
    writer.add_uint32("llama.block_count", config.num_hidden_layers)
    writer.add_uint32("llama.feed_forward_length", config.intermediate_size)
    writer.add_uint32("llama.attention.head_count", config.num_attention_heads)
    writer.add_uint32("llama.attention.head_count_kv", config.num_key_value_heads)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_uint32("llama.vocab_size", config.vocab_size)

    # 转换张量
    ggml_type = GGML_TYPE_F16 if dtype == "f16" else GGML_TYPE_F32
    state_dict = model.state_dict()
    converted = 0

    for param_name, param in state_dict.items():
        gguf_name = map_weight_name(param_name)
        if gguf_name is None:
            continue

        tensor = param.detach().cpu()
        if dtype == "f16":
            np_data = tensor.to(torch.float16).numpy()
        else:
            np_data = tensor.to(torch.float32).numpy()

        writer.add_tensor(gguf_name, np_data, ggml_type)
        converted += 1

    print(f"张量转换: {converted} 个")
    writer.write()


def main():
    parser = argparse.ArgumentParser(description="导出 ClearMind-HF 为 GGUF")
    parser.add_argument("--model", type=str, required=True, help="HF 格式模型目录")
    parser.add_argument("--output", type=str, default="model.gguf")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f32", "f16"])
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    model = ClearMindForCausalLM.from_pretrained(args.model)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count / 1e6:.1f}M, 精度: {args.dtype}")

    export_gguf(model, args.output, dtype=args.dtype)
    print(f"\n使用方式: llama-cli -m {args.output} -p 'Hello'")


if __name__ == "__main__":
    main()

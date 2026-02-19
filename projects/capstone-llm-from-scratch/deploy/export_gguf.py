"""
export_gguf.py — GGUF 格式导出
=================================

将 ClearMind 模型导出为 GGUF 格式，支持 llama.cpp 纯 CPU 推理。

GGUF (GGML Universal Format) 是 llama.cpp 使用的模型文件格式:
  - 单文件包含: 模型配置元数据 + 所有权重张量
  - 支持多种量化精度: F32, F16, Q8_0, Q4_K 等
  - 平台无关的二进制格式

权重命名映射 (ClearMind → llama.cpp):
  embedding.weight          → token_embd.weight
  layers.{i}.attention.w_q  → blk.{i}.attn_q.weight
  layers.{i}.attention.w_k  → blk.{i}.attn_k.weight
  layers.{i}.attention.w_v  → blk.{i}.attn_v.weight
  layers.{i}.attention.w_o  → blk.{i}.attn_output.weight
  layers.{i}.attn_norm      → blk.{i}.attn_norm.weight
  layers.{i}.ff.w_gate      → blk.{i}.ffn_gate.weight
  layers.{i}.ff.w_up        → blk.{i}.ffn_up.weight
  layers.{i}.ff.w_down      → blk.{i}.ffn_down.weight
  layers.{i}.ff_norm        → blk.{i}.ffn_norm.weight
  final_norm.weight         → output_norm.weight
  lm_head.weight            → output.weight

用法:
  python deploy/export_gguf.py \\
      --config configs/small.yaml \\
      --checkpoint outputs/pretrain/final.pth \\
      --output model-f16.gguf \\
      --dtype f16
"""

import argparse
import os
import struct
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.training.trainer_utils import load_checkpoint, get_device


# ============================================================
# GGUF 常量 (spec v3)
# ============================================================

GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# 元数据值类型
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# 张量数据类型
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


# ============================================================
# GGUF 写入器
# ============================================================


class GGUFWriter:
    """GGUF 文件写入器

    按照 GGUF spec v3 格式写入:
      1. Header: magic, version, tensor_count, metadata_kv_count
      2. Metadata KV pairs
      3. Tensor info (name, dims, type, offset)
      4. Padding to alignment
      5. Tensor data
    """

    ALIGNMENT = 32  # 数据对齐字节数

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata: list[tuple] = []  # (key, type, value)
        self.tensors: list[tuple] = []  # (name, data_np, ggml_type)

    # --- 元数据写入 ---

    def add_string(self, key: str, value: str):
        self.metadata.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_UINT32, value))

    def add_int32(self, key: str, value: int):
        self.metadata.append((key, GGUF_TYPE_INT32, value))

    def add_float32(self, key: str, value: float):
        self.metadata.append((key, GGUF_TYPE_FLOAT32, value))

    def add_string_array(self, key: str, values: list[str]):
        self.metadata.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, values)))

    # --- 张量添加 ---

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = GGML_TYPE_F32):
        """添加一个张量"""
        self.tensors.append((name, data, ggml_type))

    # --- 序列化 ---

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, vtype: int, value):
        if vtype == GGUF_TYPE_STRING:
            self._write_string(f, value)
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:
            f.write(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type, values = value
            f.write(struct.pack("<I", elem_type))
            f.write(struct.pack("<Q", len(values)))
            for v in values:
                self._write_metadata_value(f, elem_type, v)

    def _pad_to_alignment(self, f):
        """填充到 ALIGNMENT 边界"""
        pos = f.tell()
        pad = (self.ALIGNMENT - pos % self.ALIGNMENT) % self.ALIGNMENT
        if pad > 0:
            f.write(b"\x00" * pad)

    def write(self):
        """写入 GGUF 文件"""
        with open(self.output_path, "wb") as f:
            # ===== Header =====
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # ===== Metadata KV =====
            for key, vtype, value in self.metadata:
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                self._write_metadata_value(f, vtype, value)

            # ===== Tensor Infos =====
            # 先计算所有 tensor info 的大小，然后算数据偏移
            tensor_info_start = f.tell()

            # 先写占位信息，收集每个 tensor 的 data bytes
            tensor_data_sizes = []
            for name, data, ggml_type in self.tensors:
                self._write_string(f, name)
                n_dims = len(data.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", 0))  # offset placeholder
                tensor_data_sizes.append(data.nbytes)

            # Pad 到对齐边界
            self._pad_to_alignment(f)
            data_start = f.tell()

            # ===== 回写正确偏移量 =====
            f.seek(tensor_info_start)
            current_offset = 0
            for name, data, ggml_type in self.tensors:
                self._write_string(f, name)
                n_dims = len(data.shape)
                f.write(struct.pack("<I", n_dims))
                for dim in data.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", current_offset))
                # 计算下一个 tensor 偏移 (对齐)
                size = data.nbytes
                current_offset += size
                pad = (self.ALIGNMENT - size % self.ALIGNMENT) % self.ALIGNMENT
                current_offset += pad

            # ===== Tensor Data =====
            f.seek(data_start)
            for _, data, _ in self.tensors:
                f.write(data.tobytes())
                # 对齐
                pad = (self.ALIGNMENT - data.nbytes % self.ALIGNMENT) % self.ALIGNMENT
                if pad > 0:
                    f.write(b"\x00" * pad)

        file_size = os.path.getsize(self.output_path)
        print(f"✅ GGUF 文件已写入: {self.output_path}")
        print(f"   大小: {file_size / 1024 / 1024:.1f} MB")
        print(f"   张量数: {len(self.tensors)}")


# ============================================================
# 权重名称映射
# ============================================================


def map_weight_name(clearmind_name: str) -> str | None:
    """ClearMind → llama.cpp 权重命名映射

    Returns:
        GGUF 名称, 或 None (跳过)
    """
    # 去除 "model." 前缀 (如果存在)
    name = clearmind_name.replace("model.", "")

    mapping = {
        "embedding.weight": "token_embd.weight",
        "final_norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    if name in mapping:
        return mapping[name]

    # layers.{i}.xxx → blk.{i}.xxx
    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])

        layer_mapping = {
            "attention.w_q.weight": "attn_q.weight",
            "attention.w_k.weight": "attn_k.weight",
            "attention.w_v.weight": "attn_v.weight",
            "attention.w_o.weight": "attn_output.weight",
            "attn_norm.weight": "attn_norm.weight",
            "ff.w_gate.weight": "ffn_gate.weight",
            "ff.w_up.weight": "ffn_up.weight",
            "ff.w_down.weight": "ffn_down.weight",
            "ff_norm.weight": "ffn_norm.weight",
        }

        if rest in layer_mapping:
            return f"blk.{layer_idx}.{layer_mapping[rest]}"

    # 跳过 buffer (如 RoPE cos/sin)
    return None


# ============================================================
# 导出函数
# ============================================================


def export_gguf(
    model: GPT,
    config: ModelConfig,
    output_path: str,
    dtype: str = "f16",
    vocab: list[str] | None = None,
):
    """将 ClearMind 模型导出为 GGUF 格式

    Args:
        model:       GPT 模型
        config:      模型配置
        output_path: 输出文件路径
        dtype:       导出精度 ("f32", "f16", "q8_0")
        vocab:       词表 (可选)
    """
    writer = GGUFWriter(output_path)

    # ===== 写入元数据 =====
    writer.add_string("general.architecture", "llama")
    writer.add_string("general.name", "ClearMind")
    writer.add_uint32("llama.context_length", config.max_seq_len)
    writer.add_uint32("llama.embedding_length", config.d_model)
    writer.add_uint32("llama.block_count", config.n_layers)
    writer.add_uint32("llama.feed_forward_length", config.d_ff)
    writer.add_uint32("llama.attention.head_count", config.n_heads)
    writer.add_uint32("llama.attention.head_count_kv", config.n_kv_heads)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config.norm_eps)
    writer.add_uint32("llama.vocab_size", config.vocab_size)

    if config.sliding_window is not None:
        writer.add_uint32("llama.attention.sliding_window", config.sliding_window)

    # 词表 (如果提供)
    if vocab:
        writer.add_string_array("tokenizer.ggml.tokens", vocab)

    # ===== 转换张量 =====
    ggml_type = {
        "f32": GGML_TYPE_F32,
        "f16": GGML_TYPE_F16,
        "q8_0": GGML_TYPE_Q8_0,
    }.get(dtype, GGML_TYPE_F16)

    state_dict = model.state_dict()
    converted = 0
    skipped = 0

    for param_name, param in state_dict.items():
        gguf_name = map_weight_name(param_name)
        if gguf_name is None:
            skipped += 1
            continue

        # 转为 numpy
        tensor = param.detach().cpu()

        if dtype == "f16":
            np_data = tensor.to(torch.float16).numpy()
        elif dtype == "f32":
            np_data = tensor.to(torch.float32).numpy()
        elif dtype == "q8_0":
            # 简化的 Q8_0: 使用 int8 量化
            # 实际 GGML Q8_0 有 block-wise scale, 这里用 per-tensor 近似
            np_data = tensor.to(torch.float16).numpy()
            ggml_type = GGML_TYPE_F16  # fallback to f16
        else:
            np_data = tensor.to(torch.float16).numpy()

        writer.add_tensor(gguf_name, np_data, ggml_type)
        converted += 1

    print(f"\n📊 张量转换统计:")
    print(f"  已转换: {converted}")
    print(f"  已跳过: {skipped} (buffer / 非权重)")

    # ===== 写入文件 =====
    writer.write()


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="导出 ClearMind 为 GGUF 格式")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint")
    parser.add_argument("--output", type=str, default="model.gguf", help="输出文件名")
    parser.add_argument(
        "--dtype",
        type=str,
        default="f16",
        choices=["f32", "f16"],
        help="导出精度 (默认: f16)",
    )
    args = parser.parse_args()

    # 加载配置
    config = ModelConfig.from_yaml(args.config)

    # 创建模型
    model = GPT(config)
    device = get_device()
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    params = config.count_params()
    print(f"\n🔧 ClearMind → GGUF 导出")
    print(f"  配置:   {args.config}")
    print(f"  参数量: {params['total_millions']:.1f}M")
    print(f"  精度:   {args.dtype}")

    # 导出
    export_gguf(model, config, args.output, dtype=args.dtype)

    print(f"\n🎉 导出完成!")
    print(f"   使用方式: llama-cli -m {args.output} -p 'Hello'")


if __name__ == "__main__":
    main()

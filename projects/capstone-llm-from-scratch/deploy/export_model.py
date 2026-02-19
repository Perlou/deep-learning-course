"""
export_model.py — 模型导出与优化
==================================

为生产部署提供模型优化工具:
  1. 权重瘦身:     只保存模型权重，去除优化器状态
  2. TorchScript:  导出为独立可加载的 TorchScript 模型
  3. INT8 量化:    动态量化，减少推理显存和延迟

使用方法:
  # 导出精简权重 (只保存 state_dict)
  python deploy/export_model.py --model outputs/dpo/final.pth --format weights

  # 导出 TorchScript (可在无 Python 环境下加载)
  python deploy/export_model.py --model outputs/dpo/final.pth --format torchscript

  # INT8 动态量化 (减少模型大小和推理延迟)
  python deploy/export_model.py --model outputs/dpo/final.pth --format quantized

  # 导出所有格式
  python deploy/export_model.py --model outputs/dpo/final.pth --format all
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.data.tokenizer import ClearMindTokenizer
from src.training.trainer_utils import get_device, load_checkpoint


def get_model_size(path: str) -> str:
    """获取文件大小的可读格式"""
    size = os.path.getsize(path)
    if size > 1024 * 1024 * 1024:
        return f"{size / (1024**3):.2f} GB"
    elif size > 1024 * 1024:
        return f"{size / (1024**2):.1f} MB"
    else:
        return f"{size / 1024:.1f} KB"


def export_weights(model, output_path: str, model_config: ModelConfig):
    """导出精简权重 (只保存 state_dict + config)

    去除优化器状态和训练元数据，大幅减小文件体积。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    export_data = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": model_config.d_model,
            "n_heads": model_config.n_heads,
            "n_kv_heads": model_config.n_kv_heads,
            "n_layers": model_config.n_layers,
            "d_ff": model_config.d_ff,
            "vocab_size": model_config.vocab_size,
            "max_seq_len": model_config.max_seq_len,
            "dropout": model_config.dropout,
            "norm_eps": model_config.norm_eps,
        },
        "format": "clearmind_weights_v1",
    }

    torch.save(export_data, output_path)
    print(f"  ✅ 精简权重: {output_path} ({get_model_size(output_path)})")


def export_torchscript(model, output_path: str, model_config: ModelConfig):
    """导出 TorchScript 模型

    TorchScript 模型可以在没有 Python 环境的情况下加载和运行,
    适合嵌入到 C++ 应用或移动端。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    # 创建示例输入
    dummy_input = torch.randint(0, model_config.vocab_size, (1, 32))

    try:
        # 使用 trace 方式导出
        traced_model = torch.jit.trace(model_cpu, dummy_input)
        traced_model.save(output_path)
        print(f"  ✅ TorchScript: {output_path} ({get_model_size(output_path)})")
    except Exception as e:
        print(f"  ⚠️  TorchScript 导出失败: {e}")
        print("     (某些动态操作可能不支持 trace 模式)")

        # 尝试 script 方式
        try:
            scripted_model = torch.jit.script(model_cpu)
            scripted_model.save(output_path)
            print(
                f"  ✅ TorchScript (script): {output_path} ({get_model_size(output_path)})"
            )
        except Exception as e2:
            print(f"  ❌ TorchScript 导出均失败: {e2}")

    # 恢复模型到原设备
    device = get_device()
    model.to(device)


def export_quantized(model, output_path: str, model_config: ModelConfig):
    """INT8 动态量化

    对 Linear 层进行动态量化:
    - 权重: 静态量化为 INT8
    - 激活: 动态量化 (运行时计算 scale/zero_point)

    效果:
    - 模型大小减少 ~2-4x
    - CPU 推理速度提升 ~1.5-2x
    - 精度损失通常很小
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    # 动态量化 Linear 层
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # 保存量化后的权重
    export_data = {
        "model_state_dict": quantized_model.state_dict(),
        "config": {
            "d_model": model_config.d_model,
            "n_heads": model_config.n_heads,
            "n_kv_heads": model_config.n_kv_heads,
            "n_layers": model_config.n_layers,
            "d_ff": model_config.d_ff,
            "vocab_size": model_config.vocab_size,
            "max_seq_len": model_config.max_seq_len,
            "dropout": model_config.dropout,
            "norm_eps": model_config.norm_eps,
        },
        "format": "clearmind_quantized_int8_v1",
        "quantization": "dynamic_int8",
    }

    torch.save(export_data, output_path)
    print(f"  ✅ 量化模型: {output_path} ({get_model_size(output_path)})")

    # 恢复模型到原设备
    device = get_device()
    model.to(device)


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="ClearMind 模型导出")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/dpo/final.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="outputs/tokenizer/tokenizer.model"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="weights",
        choices=["weights", "torchscript", "quantized", "gguf", "all"],
        help="导出格式",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/export",
        help="导出目录",
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(**config["model"])

    # 加载分词器 (更新 vocab_size)
    if os.path.exists(args.tokenizer):
        tok = ClearMindTokenizer(args.tokenizer)
        if tok.vocab_size != model_config.vocab_size:
            model_config.vocab_size = tok.vocab_size

    # 加载模型
    device = get_device()
    model = GPT(model_config).to(device)

    if not os.path.exists(args.model):
        for path in [
            "outputs/dpo/final.pth",
            "outputs/sft/final.pth",
            "outputs/pretrain/final.pth",
        ]:
            if os.path.exists(path):
                args.model = path
                break
        else:
            print("❌ 未找到任何 checkpoint")
            sys.exit(1)

    load_checkpoint(model, args.model, device=device)
    model.eval()

    param_info = model.count_parameters()
    original_size = get_model_size(args.model)

    print(f"\n{'=' * 60}")
    print(f"📦 ClearMind 模型导出")
    print(f"{'=' * 60}")
    print(f"  源模型:   {args.model} ({original_size})")
    print(f"  参数量:   {param_info['total_millions']:.1f}M")
    print(f"  导出格式: {args.format}")
    print(f"  输出目录: {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    formats_to_export = (
        ["weights", "torchscript", "quantized", "gguf"]
        if args.format == "all"
        else [args.format]
    )

    for fmt in formats_to_export:
        if fmt == "weights":
            export_weights(
                model,
                os.path.join(args.output_dir, "clearmind_weights.pth"),
                model_config,
            )
        elif fmt == "torchscript":
            export_torchscript(
                model,
                os.path.join(args.output_dir, "clearmind_scripted.pt"),
                model_config,
            )
        elif fmt == "quantized":
            export_quantized(
                model,
                os.path.join(args.output_dir, "clearmind_int8.pth"),
                model_config,
            )
        elif fmt == "gguf":
            from deploy.export_gguf import export_gguf

            export_gguf(
                model,
                model_config,
                os.path.join(args.output_dir, "clearmind-f16.gguf"),
                dtype="f16",
            )

    print(f"\n{'=' * 60}")
    print(f"✅ 导出完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
export_model.py — 模型导出与优化 (HF 版)
==========================================

from-scratch 对比:
  - from-scratch: GPT + load_checkpoint + torch.save 手动导出
  - HF 版: from_pretrained + save_pretrained + 量化

导出格式:
  1. HF 标准格式:  save_pretrained (config.json + model.safetensors)
  2. INT8 量化:     torch.quantization.quantize_dynamic
  3. GGUF:          调用 export_gguf.py

用法:
  python deploy/export_model.py --model outputs/sft --format hf
  python deploy/export_model.py --model outputs/sft --format quantized
  python deploy/export_model.py --model outputs/sft --format gguf
  python deploy/export_model.py --model outputs/sft --format all
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from model import ClearMindForCausalLM


def get_model_size(path: str) -> str:
    """获取文件/目录大小"""
    if os.path.isfile(path):
        size = os.path.getsize(path)
    else:
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    if size > 1024 * 1024 * 1024:
        return f"{size / (1024**3):.2f} GB"
    elif size > 1024 * 1024:
        return f"{size / (1024**2):.1f} MB"
    return f"{size / 1024:.1f} KB"


def export_hf(model, tokenizer_path: str | None, output_dir: str):
    """导出 HF 标准格式

    from-scratch 对比:
      - from-scratch: torch.save(state_dict) 手动保存
      - HF 版: save_pretrained() 自动保存 config + weights + tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # 复制 tokenizer (如果提供)
    if tokenizer_path and os.path.exists(tokenizer_path):
        from data.tokenizer import ClearMindTokenizer
        tokenizer = ClearMindTokenizer.load(tokenizer_path)
        tokenizer.save_pretrained(output_dir)

    print(f"  HF 格式: {output_dir} ({get_model_size(output_dir)})")


def export_quantized(model, output_dir: str):
    """INT8 动态量化

    对 Linear 层进行动态量化，减少模型大小和 CPU 推理延迟。
    """
    os.makedirs(output_dir, exist_ok=True)

    model_cpu = model.cpu()
    model_cpu.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8,
    )

    # 保存量化后的 state_dict
    output_path = os.path.join(output_dir, "clearmind_int8.pth")
    torch.save({
        "model_state_dict": quantized_model.state_dict(),
        "config": model.config.to_dict(),
        "quantization": "dynamic_int8",
    }, output_path)

    print(f"  量化模型: {output_path} ({get_model_size(output_path)})")


def export_gguf_format(model, output_dir: str):
    """GGUF 格式导出"""
    from deploy.export_gguf import export_gguf as _export_gguf

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clearmind-f16.gguf")
    _export_gguf(model, output_path, dtype="f16")


def main():
    parser = argparse.ArgumentParser(description="ClearMind 模型导出 (HF 版)")
    parser.add_argument("--model", type=str, required=True, help="HF 格式模型目录")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer 目录")
    parser.add_argument("--format", type=str, default="hf",
                        choices=["hf", "quantized", "gguf", "all"])
    parser.add_argument("--output_dir", type=str, default="outputs/export")
    args = parser.parse_args()

    print(f"加载模型: {args.model}")
    model = ClearMindForCausalLM.from_pretrained(args.model)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count / 1e6:.1f}M")
    print(f"导出格式: {args.format}\n")

    formats = ["hf", "quantized", "gguf"] if args.format == "all" else [args.format]

    for fmt in formats:
        if fmt == "hf":
            export_hf(model, args.tokenizer, os.path.join(args.output_dir, "hf"))
        elif fmt == "quantized":
            export_quantized(model, os.path.join(args.output_dir, "quantized"))
        elif fmt == "gguf":
            export_gguf_format(model, os.path.join(args.output_dir, "gguf"))

    print(f"\n导出完成!")


if __name__ == "__main__":
    main()

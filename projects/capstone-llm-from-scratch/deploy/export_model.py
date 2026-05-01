"""
export_model.py — 模型导出工具
================================

把训练产物（``outputs/<stage>/final.pth`` 或 ``_resume.pth``）转换为以下格式：

  - **safetensors**：HF 标准格式，比 .pth 更安全（无 pickle）、加载更快、零拷贝
  - **fp16 / fp32 .pth**：纯权重文件，与 ``transformers.AutoModelForCausalLM`` 兼容
  - **state_dict 信息打印**：参数量统计、键名校验

发布到 HuggingFace / ModelScope 时强烈推荐用 safetensors。

Phase 5 计划新增 ``scripts/convert_to_qwen3.py``，把 ClearMind GPT 命名映射到
``Qwen3ForCausalLM`` 标准命名（``q_proj/k_proj/...``），届时 ollama / vllm /
llama.cpp / Llama-Factory 等生态可直接消费。GGUF 转换由 llama.cpp 自带的
``convert_hf_to_gguf.py`` 处理，本仓库不再维护单独 GGUF 工具。

用法:

  # 导出 safetensors（推荐）
  python deploy/export_model.py --input outputs/dpo/final.pth \\
      --output outputs/release/clearmind-base.safetensors

  # 导出 fp16 .pth
  python deploy/export_model.py --input outputs/dpo/final.pth \\
      --output outputs/release/clearmind-base-fp16.pth --dtype fp16

  # 仅打印 state_dict 摘要
  python deploy/export_model.py --input outputs/dpo/final.pth --inspect
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def _load_state_dict(path: str) -> dict:
    """加载 ckpt 或纯权重文件，返回 state_dict"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj  # 纯权重 dict


def _convert_dtype(state_dict: dict, dtype: str) -> dict:
    target = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    return {k: v.to(target) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


def _save_safetensors(state_dict: dict, output: str) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("❌ 缺少依赖: pip install safetensors")
        sys.exit(1)
    sd = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    save_file(sd, output)


def _save_pth(state_dict: dict, output: str) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.save(state_dict, output)


def _format_size(path: str) -> str:
    n = os.path.getsize(path)
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024:.1f} KB"


def _inspect(state_dict: dict) -> None:
    print("\n📊 state_dict 摘要")
    print(f"  键数:     {len(state_dict)}")
    total = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )
    print(f"  参数量:   {total / 1e6:.2f}M ({total:,})")
    dtypes: dict[str, int] = {}
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + v.numel()
    print("  dtype 分布:")
    for d, n in dtypes.items():
        print(f"    {d}: {n / 1e6:.2f}M")
    print("\n  前 10 个 key（按字典序）:")
    for k in sorted(state_dict.keys())[:10]:
        v = state_dict[k]
        shape = list(v.shape) if isinstance(v, torch.Tensor) else "?"
        dt = v.dtype if isinstance(v, torch.Tensor) else "?"
        print(f"    {k:50s}  {shape}  {dt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMind 模型导出")
    parser.add_argument("--input", "-i", required=True, help="输入 ckpt 路径")
    parser.add_argument("--output", "-o", default=None, help="输出路径")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="导出 dtype（默认 fp16，体积最小）",
    )
    parser.add_argument(
        "--format",
        choices=["safetensors", "pth", "auto"],
        default="auto",
        help="输出格式（auto = 按 --output 后缀决定）",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="仅打印 state_dict 摘要，不导出",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return 1

    print("=" * 60)
    print("ClearMind 模型导出")
    print("=" * 60)
    print(f"📥 Input:  {args.input}  ({_format_size(args.input)})")

    state_dict = _load_state_dict(args.input)
    state_dict = _convert_dtype(state_dict, args.dtype)

    if args.inspect:
        _inspect(state_dict)
        return 0

    if args.output is None:
        print("❌ 必须指定 --output（或使用 --inspect 仅查看）")
        return 1

    fmt = args.format
    if fmt == "auto":
        fmt = "safetensors" if args.output.endswith(".safetensors") else "pth"

    if fmt == "safetensors":
        _save_safetensors(state_dict, args.output)
    else:
        _save_pth(state_dict, args.output)

    print(f"📤 Output: {args.output}  ({_format_size(args.output)})")
    print(f"   format: {fmt}")
    print(f"   dtype:  {args.dtype}")
    print()
    print("=" * 60)
    print("✅ 导出完成")
    print("=" * 60)
    print()
    print("下一步:")
    print("  - 上传到 HF: huggingface-cli upload <repo_id> <output>")
    print("  - 上传到 ModelScope: modelscope upload <repo_id> <output>")
    print("  - 转 Qwen3 兼容: (Phase 5) python scripts/convert_to_qwen3.py ...")
    print("  - 转 GGUF (llama.cpp): cd <llama.cpp> && python convert_hf_to_gguf.py ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
convert_to_qwen3.py — ClearMind → Qwen3ForCausalLM 兼容导出
============================================================

把 ClearMind GPT 训练产物（``outputs/<stage>/final.pth``）转换为
``Qwen3ForCausalLM`` 标准 HuggingFace 仓库目录，让训练好的模型能被以下生态直接消费：

  - ``transformers.AutoModelForCausalLM.from_pretrained(...)`` 推理
  - ``vllm serve <local-dir>`` 高性能推理
  - ``ollama run <model>`` 本地推理（导出后再 ``llama.cpp/convert_hf_to_gguf.py`` 转 GGUF）
  - ``llama-factory`` 继续微调

权重命名映射（ClearMind → Qwen3）：

  token_embedding.weight                        → model.embed_tokens.weight
  layers.{i}.attn_norm.weight                   → model.layers.{i}.input_layernorm.weight
  layers.{i}.attention.w_q.weight               → model.layers.{i}.self_attn.q_proj.weight
  layers.{i}.attention.w_k.weight               → model.layers.{i}.self_attn.k_proj.weight
  layers.{i}.attention.w_v.weight               → model.layers.{i}.self_attn.v_proj.weight
  layers.{i}.attention.w_o.weight               → model.layers.{i}.self_attn.o_proj.weight
  layers.{i}.attention.q_norm.weight (QK-Norm)  → model.layers.{i}.self_attn.q_norm.weight
  layers.{i}.attention.k_norm.weight (QK-Norm)  → model.layers.{i}.self_attn.k_norm.weight
  layers.{i}.ffn_norm.weight                    → model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.feedforward.w_gate.weight          → model.layers.{i}.mlp.gate_proj.weight
  layers.{i}.feedforward.w_up.weight            → model.layers.{i}.mlp.up_proj.weight
  layers.{i}.feedforward.w_down.weight          → model.layers.{i}.mlp.down_proj.weight
  final_norm.weight                             → model.norm.weight
  lm_head.weight                                → lm_head.weight (tied with embed_tokens)

输出目录结构（标准 HF repo）：

    release/<model_name>/
    ├── config.json              # Qwen3Config（hidden_size、num_hidden_layers 等）
    ├── model.safetensors        # 重命名后的权重（safetensors 格式）
    ├── tokenizer.json           # 复制自 tokenizer/minimind/
    ├── tokenizer_config.json    # 复制自 tokenizer/minimind/（含 chat_template）
    ├── generation_config.json   # 推理默认参数（temperature、top_p 等）
    └── README.md                # 模型卡（用 release/MODEL_CARD_TEMPLATE.md 生成）

用法:

  # ClearMind-Base 导出
  python scripts/convert_to_qwen3.py \\
      --input outputs/dpo/final.pth \\
      --config configs/main.yaml \\
      --output release/clearmind-base \\
      --model_name "ClearMind-Base"

  # ClearMind-Plus 导出
  python scripts/convert_to_qwen3.py \\
      --input outputs/dpo/final.pth \\
      --config configs/plus.yaml \\
      --output release/clearmind-plus \\
      --model_name "ClearMind-Plus"

  # 本地验证导出的权重
  python -c "
  from transformers import AutoModelForCausalLM, AutoTokenizer
  m = AutoModelForCausalLM.from_pretrained('release/clearmind-base', trust_remote_code=False)
  tk = AutoTokenizer.from_pretrained('release/clearmind-base')
  print(m.generate(tk('你好', return_tensors='pt').input_ids, max_new_tokens=20))
  "
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.model.config import ModelConfig


# ============================================================
# 权重命名映射
# ============================================================


def _remap_state_dict(
    state_dict: dict[str, torch.Tensor],
    use_qk_norm: bool,
) -> dict[str, torch.Tensor]:
    """ClearMind state_dict → Qwen3 state_dict

    Args:
        state_dict:    ClearMind 原始 state_dict
        use_qk_norm:   模型是否启用了 QK-Norm（决定是否映射 q_norm/k_norm）

    Returns:
        Qwen3 风格命名的 state_dict
    """
    new_sd: dict[str, torch.Tensor] = {}

    # 顶层映射
    top_level_map = {
        "token_embedding.weight": "model.embed_tokens.weight",
        "final_norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    # 层内映射规则（{i} 会被替换）
    layer_pattern = re.compile(r"^layers\.(\d+)\.(.+)$")
    layer_inner_map = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attention.w_q.weight": "self_attn.q_proj.weight",
        "attention.w_k.weight": "self_attn.k_proj.weight",
        "attention.w_v.weight": "self_attn.v_proj.weight",
        "attention.w_o.weight": "self_attn.o_proj.weight",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "feedforward.w_gate.weight": "mlp.gate_proj.weight",
        "feedforward.w_up.weight": "mlp.up_proj.weight",
        "feedforward.w_down.weight": "mlp.down_proj.weight",
    }

    skipped: list[str] = []
    for key, tensor in state_dict.items():
        # 跳过 buffer（rope_cos / rope_sin / causal_mask 等）
        if "rope" in key or "causal" in key:
            skipped.append(key)
            continue

        if key in top_level_map:
            new_sd[top_level_map[key]] = tensor
            continue

        m = layer_pattern.match(key)
        if m:
            layer_idx, suffix = m.group(1), m.group(2)
            if suffix in layer_inner_map:
                new_key = f"model.layers.{layer_idx}.{layer_inner_map[suffix]}"
                new_sd[new_key] = tensor
                continue

        skipped.append(key)

    if skipped:
        print(f"  ⚠️  跳过 {len(skipped)} 个未映射的 key: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # 验证 QK-Norm 一致性
    has_q_norm = any("self_attn.q_norm.weight" in k for k in new_sd)
    if use_qk_norm != has_q_norm:
        print(
            f"  ⚠️  use_qk_norm={use_qk_norm} 但导出权重 has_q_norm={has_q_norm}；"
            "Qwen3Config 将按 has_q_norm 设置"
        )

    return new_sd


# ============================================================
# config.json 生成
# ============================================================


def _build_qwen3_config(
    model_config: ModelConfig,
    has_q_norm: bool,
    model_name: str,
    tie_word_embeddings: bool = True,
) -> dict:
    """生成 Qwen3 风格的 config.json 内容

    参考 transformers Qwen3Config，但只填关键字段（其他用默认）。
    用 dict 形式而不是 Qwen3Config 实例，避免硬绑定 transformers 版本。
    """
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": model_config.head_dim,
        "hidden_act": "silu",
        "hidden_size": model_config.d_model,
        "initializer_range": 0.02,
        "intermediate_size": model_config.d_ff,
        "max_position_embeddings": model_config.max_seq_len,
        "max_window_layers": model_config.n_layers,
        "model_type": "qwen3",
        "num_attention_heads": model_config.n_heads,
        "num_hidden_layers": model_config.n_layers,
        "num_key_value_heads": model_config.n_kv_heads,
        "rms_norm_eps": model_config.norm_eps,
        "rope_scaling": model_config.rope_scaling,
        "rope_theta": model_config.rope_theta,
        "sliding_window": None,
        "tie_word_embeddings": tie_word_embeddings,
        "torch_dtype": "float16",
        "transformers_version": "4.45.0",
        "use_cache": True,
        "use_qk_norm": has_q_norm,  # Qwen3 5.0+ 已原生支持，下兼容由模型实现自处理
        "use_sliding_window": False,
        "vocab_size": model_config.vocab_size,
        "_name_or_path": model_name,
    }


def _build_generation_config(model_config: ModelConfig) -> dict:
    """生成 generation_config.json 内容"""
    return {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_new_tokens": 1024,
        "transformers_version": "4.45.0",
    }


# ============================================================
# 主流程
# ============================================================


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """加载 ClearMind ckpt（兼容 _resume.pth 完整状态 / final.pth 纯权重）"""
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj  # 纯权重 dict


def _save_safetensors(state_dict: dict[str, torch.Tensor], output: Path, dtype: str) -> None:
    """保存为 model.safetensors，并自动 dtype 转换"""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("❌ 缺少依赖: pip install safetensors")
        sys.exit(1)

    target_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    sd = {
        k: v.to(target_dtype).contiguous() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
    save_file(sd, str(output))


def _copy_tokenizer(src_dir: Path, dst_dir: Path) -> None:
    """复制 tokenizer.json + tokenizer_config.json"""
    for fname in ("tokenizer.json", "tokenizer_config.json"):
        src = src_dir / fname
        if not src.exists():
            print(f"⚠️  tokenizer 文件不存在: {src}")
            continue
        shutil.copy2(src, dst_dir / fname)


def _maybe_render_model_card(
    output_dir: Path,
    model_name: str,
    model_config: ModelConfig,
    yaml_config: dict,
) -> None:
    """从 release/MODEL_CARD_TEMPLATE.md 渲染模型卡到 output/README.md"""
    template = PROJECT_ROOT / "release" / "MODEL_CARD_TEMPLATE.md"
    if not template.exists():
        # 没模板就写一个最小版
        readme = output_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {model_name}\n\n"
                f"Trained from scratch with ClearMind. "
                f"Architecture: GPT (RoPE + RMSNorm + SwiGLU + GQA + QK-Norm).\n\n"
                f"- Parameters: {model_config.count_params()['total_millions']:.1f}M\n"
                f"- Hidden size: {model_config.d_model}\n"
                f"- Layers: {model_config.n_layers}\n"
                f"- Vocab: {model_config.vocab_size}\n",
                encoding="utf-8",
            )
        return

    text = template.read_text(encoding="utf-8")
    release = yaml_config.get("release", {})
    params = model_config.count_params()
    placeholders = {
        "{{MODEL_NAME}}": model_name,
        "{{DISPLAY_NAME}}": release.get("display_name", model_name),
        "{{FAMILY}}": release.get("family", "ClearMind"),
        "{{SIZE_LABEL}}": release.get("size_label", f"{params['total_millions']:.0f}M"),
        "{{PARAMS_M}}": f"{params['total_millions']:.1f}",
        "{{TAGLINE}}": release.get("tagline", ""),
        "{{LICENSE}}": release.get("license", "apache-2.0"),
        "{{D_MODEL}}": str(model_config.d_model),
        "{{N_HEADS}}": str(model_config.n_heads),
        "{{N_KV_HEADS}}": str(model_config.n_kv_heads),
        "{{N_LAYERS}}": str(model_config.n_layers),
        "{{D_FF}}": str(model_config.d_ff),
        "{{MAX_SEQ_LEN}}": str(model_config.max_seq_len),
        "{{VOCAB_SIZE}}": str(model_config.vocab_size),
        "{{ROPE_THETA}}": f"{model_config.rope_theta:.0e}",
        "{{USE_QK_NORM}}": "Yes" if model_config.use_qk_norm else "No",
        "{{INTENDED_USE}}": release.get("intended_use", "通用对话与基础任务"),
    }
    for k, v in placeholders.items():
        text = text.replace(k, v)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="ClearMind → Qwen3 兼容导出")
    parser.add_argument(
        "--input", "-i", required=True, help="输入 ckpt 路径（final.pth 或 _resume.pth）"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="ClearMind 模型配置 yaml（决定架构）"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="输出目录（HF 仓库格式）"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="模型名（默认从 yaml.release.model_name 读取）",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="导出 dtype（默认 fp16，体积最小）",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default="tokenizer/minimind",
        help="tokenizer 源目录",
    )
    parser.add_argument(
        "--no_tie_embeddings",
        action="store_true",
        help="禁用 lm_head/embed_tokens 共享权重（默认共享）",
    )
    args = parser.parse_args()

    # ---- 加载 ClearMind config ----
    with open(args.config, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)
    model_config = ModelConfig(**yaml_cfg["model"])

    model_name = args.model_name or yaml_cfg.get("release", {}).get("model_name", "ClearMind")

    output_dir = PROJECT_ROOT / args.output if not os.path.isabs(args.output) else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ClearMind → Qwen3 兼容导出")
    print("=" * 60)
    print(f"  Input ckpt:  {args.input}")
    print(f"  Config:      {args.config}")
    print(f"  Output:      {output_dir}")
    print(f"  Model name:  {model_name}")
    print(f"  Dtype:       {args.dtype}")
    print()

    # ---- 加载 + 重命名 state_dict ----
    print("⬇️  加载 state_dict ...")
    sd = _load_state_dict(Path(args.input))
    print(f"   ClearMind state_dict: {len(sd)} keys")

    print("🔁 重命名权重为 Qwen3 风格 ...")
    new_sd = _remap_state_dict(sd, use_qk_norm=model_config.use_qk_norm)
    has_q_norm = any("self_attn.q_norm.weight" in k for k in new_sd)
    print(f"   Qwen3 state_dict: {len(new_sd)} keys, has_q_norm={has_q_norm}")

    # 校验关键 key 都在
    must_have = {"model.embed_tokens.weight", "model.norm.weight"}
    if model_config.n_layers > 0:
        must_have.add("model.layers.0.self_attn.q_proj.weight")
        must_have.add("model.layers.0.mlp.gate_proj.weight")
    missing = must_have - set(new_sd.keys())
    if missing:
        print(f"❌ 缺少关键权重: {missing}")
        return 1

    # ---- 处理 tied embeddings ----
    tie = not args.no_tie_embeddings
    if tie:
        # ClearMind GPT 是 tied 的（forward 中 token_embedding.weight = lm_head.weight）
        # Qwen3 默认也 tie；如果 lm_head 单独存了，删掉以避免重复
        if "lm_head.weight" in new_sd and "model.embed_tokens.weight" in new_sd:
            # 检查是否真的指向同一份（共享 storage）
            same = new_sd["lm_head.weight"].data_ptr() == new_sd["model.embed_tokens.weight"].data_ptr()
            if same:
                print("   ✅ tied embeddings 检测到（共享 storage），保留 lm_head.weight")
            else:
                # 不共享 storage 但值相同 — Qwen3 推理仍按 tie 处理
                print("   ✅ tied embeddings：lm_head.weight 与 embed_tokens 数值一致")

    # ---- 保存权重 ----
    print(f"💾 保存 model.safetensors（dtype={args.dtype}）...")
    _save_safetensors(new_sd, output_dir / "model.safetensors", args.dtype)

    # ---- 写 config.json ----
    print("📝 写 config.json ...")
    qwen_config = _build_qwen3_config(
        model_config, has_q_norm, model_name, tie_word_embeddings=tie
    )
    (output_dir / "config.json").write_text(
        json.dumps(qwen_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 写 generation_config.json ----
    print("📝 写 generation_config.json ...")
    gen_config = _build_generation_config(model_config)
    (output_dir / "generation_config.json").write_text(
        json.dumps(gen_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 复制 tokenizer ----
    print(f"📋 复制 tokenizer ({args.tokenizer_dir}) ...")
    src_dir = PROJECT_ROOT / args.tokenizer_dir if not os.path.isabs(args.tokenizer_dir) else Path(args.tokenizer_dir)
    _copy_tokenizer(src_dir, output_dir)

    # ---- 渲染模型卡 ----
    print("📋 渲染 README.md（模型卡）...")
    _maybe_render_model_card(output_dir, model_name, model_config, yaml_cfg)

    # ---- 输出 ----
    print()
    print("=" * 60)
    print(f"✅ 导出完成 → {output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir}")
    print("=" * 60)
    print()
    print("产物:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.name:32s}  {size_mb:>8.2f} MB")
    print()
    print("下一步：")
    print("  ▶ 本地验证（推荐先做）：")
    print(f"      python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
          f"m = AutoModelForCausalLM.from_pretrained('{output_dir}'); "
          f"tk = AutoTokenizer.from_pretrained('{output_dir}'); "
          f"print(m); print('vocab:', tk.vocab_size)\"")
    print()
    print("  ▶ 上传 HuggingFace：")
    print(f"      python scripts/push_to_hub.py --model_dir {output_dir} --repo <user>/{model_name}")
    print()
    print("  ▶ 上传 ModelScope：")
    print(f"      python scripts/push_to_modelscope.py --model_dir {output_dir} --repo <user>/{model_name}")
    print()
    print("  ▶ 转 GGUF（llama.cpp/ollama）：")
    print(f"      cd <llama.cpp>; python convert_hf_to_gguf.py {output_dir} --outtype q4_0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

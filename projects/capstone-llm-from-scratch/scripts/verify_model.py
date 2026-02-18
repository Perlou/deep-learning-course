"""Phase 1 验证脚本 — 测试模型架构的正确性"""

import sys
import os

# 确保能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.config import ModelConfig
from src.model.gpt import GPT


def main():
    print("=" * 60)
    print("ClearMind GPT 模型验证")
    print("=" * 60)

    # ========== Small 配置 ==========
    config = ModelConfig.small()
    model = GPT(config)
    params = model.count_parameters()

    print(f"\n📊 Small 配置:")
    print(
        f"  d_model={config.d_model}, n_heads={config.n_heads}, "
        f"n_kv_heads={config.n_kv_heads}, n_layers={config.n_layers}"
    )
    print(f"  head_dim={config.head_dim}, GQA groups={config.n_kv_groups}")
    print(f"  总参数量: {params['total_millions']:.1f}M")

    # 前向传播测试
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets)

    expected_loss = torch.tensor(config.vocab_size).float().log().item()

    print(f"\n🔄 前向传播:")
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f} (期望 ≈ {expected_loss:.4f})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        "Logits shape 错误!"
    )
    assert abs(loss.item() - expected_loss) < 1.5, "初始 loss 偏差过大!"

    # 反向传播测试
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"\n✅ 反向传播: {'成功' if grad_ok else '失败'}")

    # 生成测试
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\n✨ 文本生成:")
    print(f"  Prompt tokens:     {prompt.shape[1]}")
    print(f"  Generated tokens:  {generated.shape[1]}")
    assert generated.shape[1] == 25, "生成长度错误!"

    # YAML 配置加载测试
    config_yaml = ModelConfig.from_yaml("configs/small.yaml")
    print(f"\n📄 YAML 配置加载:")
    print(f"  d_model={config_yaml.d_model}, vocab_size={config_yaml.vocab_size}")
    assert config_yaml.d_model == 512

    # Medium 配置参数量
    config_med = ModelConfig.medium()
    model_med = GPT(config_med)
    params_med = model_med.count_parameters()
    print(f"\n📊 Medium 配置参数量: {params_med['total_millions']:.1f}M")

    print(f"\n{'=' * 60}")
    print("✅ Phase 1 全部验证通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()

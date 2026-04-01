"""LoRA 微调测试 (PEFT 版)"""

import tempfile
from pathlib import Path

import torch
import pytest
from peft import PeftModel, LoraConfig

from training.lora import create_lora_config, apply_peft_lora, merge_lora_weights, create_qlora_config
from training.sft import run_sft
from model import ClearMindConfig, ClearMindForCausalLM


# ============================================================
# TestCreateLoraConfig
# ============================================================


class TestCreateLoraConfig:
    """测试 create_lora_config"""

    def test_basic_config(self):
        """基本 LoRA 配置"""
        config = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
        lora_config = create_lora_config(config)
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert lora_config.lora_dropout == 0.05

    def test_target_modules(self):
        """target_modules 正确设置"""
        config = {"target_modules": ["q_proj", "v_proj"]}
        lora_config = create_lora_config(config)
        assert "q_proj" in lora_config.target_modules
        assert "v_proj" in lora_config.target_modules

    def test_default_values(self):
        """空配置使用合理默认值"""
        lora_config = create_lora_config({})
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
        assert lora_config.task_type.value == "CAUSAL_LM"


# ============================================================
# TestApplyPeftLora
# ============================================================


class TestApplyPeftLora:
    """测试 apply_peft_lora"""

    def test_returns_peft_model(self, tiny_config):
        """应用 LoRA 后返回 PeftModel"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)
        assert isinstance(peft_model, PeftModel)

    def test_freezes_base_params(self, tiny_config):
        """原始参数被冻结，只有 LoRA 参数可训练"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        assert trainable < total
        assert trainable > 0

    def test_forward_works(self, tiny_config):
        """LoRA 模型前向传播正常"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        output = peft_model(input_ids)
        assert output.logits.shape == (1, 8, tiny_config.vocab_size)


# ============================================================
# TestMergeLoraWeights
# ============================================================


class TestMergeLoraWeights:
    """测试 merge_lora_weights"""

    def test_merge_returns_base_model(self, tiny_config):
        """合并后不再是 PeftModel"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)
        merged = merge_lora_weights(peft_model)
        assert not isinstance(merged, PeftModel)

    def test_merge_preserves_output(self, tiny_config):
        """合并前后输出一致"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)
        peft_model.eval()

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            peft_out = peft_model(input_ids).logits

        merged = merge_lora_weights(peft_model)
        merged.eval()
        with torch.no_grad():
            merged_out = merged(input_ids).logits

        assert torch.allclose(peft_out, merged_out, atol=1e-5)


# ============================================================
# TestLoRASaveLoad
# ============================================================


class TestLoRASaveLoad:
    """测试 LoRA adapter 保存/加载"""

    def test_save_adapter(self, tiny_config):
        """LoRA adapter 可以保存"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            peft_model.save_pretrained(tmpdir)
            assert (Path(tmpdir) / "adapter_config.json").exists()
            assert (
                (Path(tmpdir) / "adapter_model.safetensors").exists()
                or (Path(tmpdir) / "adapter_model.bin").exists()
            )

    def test_load_adapter(self, tiny_config):
        """LoRA adapter 可以加载"""
        model = ClearMindForCausalLM(tiny_config)
        lora_config = create_lora_config({"r": 4, "lora_dropout": 0.0})
        peft_model = apply_peft_lora(model, lora_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            peft_model.save_pretrained(tmpdir)

            # 重新加载
            base_model = ClearMindForCausalLM(tiny_config)
            loaded = PeftModel.from_pretrained(base_model, tmpdir)
            assert isinstance(loaded, PeftModel)


# ============================================================
# TestLoRASFTIntegration
# ============================================================


class TestLoRASFTIntegration:
    """测试 LoRA + SFT 集成"""

    def test_sft_with_lora(
        self, tiny_yaml_config, sample_sft_data, saved_tmp_tokenizer, tmp_path
    ):
        """LoRA SFT 运行不报错"""
        output_dir = str(tmp_path / "lora_sft_output")
        run_sft(
            config_path=tiny_yaml_config,
            data_path=sample_sft_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
            use_lora=True,
        )
        # 合并后的模型可加载
        output_path = Path(output_dir)
        assert (output_path / "config.json").exists()
        model = ClearMindForCausalLM.from_pretrained(output_dir)
        assert model is not None


# ============================================================
# TestQLoRA
# ============================================================


class TestQLoRA:
    """测试 QLoRA 配置创建"""

    def test_create_qlora_config_no_cuda(self):
        """非 CUDA 环境下 create_qlora_config 抛出 RuntimeError"""
        import torch

        if torch.cuda.is_available():
            pytest.skip("需要非 CUDA 环境")

        with pytest.raises((RuntimeError, ImportError)):
            create_qlora_config({"r": 4})

    def test_create_qlora_config_returns_tuple_on_cuda(self):
        """CUDA 环境下返回 (BitsAndBytesConfig, LoraConfig) 元组"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA 环境")

        try:
            bnb_config, lora_config = create_qlora_config({"r": 4, "lora_alpha": 8})
            assert lora_config.r == 4
        except ImportError:
            pytest.skip("bitsandbytes 未安装")

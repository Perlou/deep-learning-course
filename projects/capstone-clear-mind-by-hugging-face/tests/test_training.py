"""训练模块测试 (HF Trainer 版)"""

import os
from pathlib import Path

import pytest
from transformers import TrainerCallback, TrainingArguments

from training.pretrain import create_training_args, run_pretrain
from training.sft import create_sft_args, run_sft
from training.dpo import create_dpo_args, run_dpo
from training.callbacks import ClearMindLoggingCallback
from model import ClearMindForCausalLM


# ============================================================
# TestCreateTrainingArgs
# ============================================================


class TestCreateTrainingArgs:
    """测试 create_training_args 函数"""

    def test_basic_fields(self, tmp_path):
        """基础字段正确透传"""
        config = {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_steps": 100,
            "learning_rate": 3e-4,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 10,
            "logging_steps": 5,
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "save_steps": 50,
            "eval_steps": 50,
            "bf16": False,
            "report_to": "none",
            "gradient_checkpointing": False,
        }
        args = create_training_args(config, str(tmp_path / "output"))
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 2
        assert args.max_steps == 100
        assert args.learning_rate == 3e-4
        assert args.warmup_steps == 10

    def test_max_steps_override(self, tmp_path):
        """max_steps 可通过 config 覆盖"""
        config = {
            "max_steps": 50,
            "per_device_train_batch_size": 2,
            "report_to": "none",
        }
        args = create_training_args(config, str(tmp_path / "output"))
        assert args.max_steps == 50

    def test_output_dir(self, tmp_path):
        """output_dir 正确设置"""
        output = str(tmp_path / "my_output")
        config = {"max_steps": 10, "report_to": "none"}
        args = create_training_args(config, output)
        assert args.output_dir == output

    def test_small_max_steps_disables_eval(self, tmp_path):
        """max_steps <= 10 时自动关闭 eval 和 save"""
        config = {
            "max_steps": 5,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "report_to": "none",
        }
        args = create_training_args(config, str(tmp_path / "output"))
        assert args.eval_strategy.value == "no"
        assert args.save_strategy.value == "no"


# ============================================================
# TestRunPretrain
# ============================================================


class TestRunPretrain:
    """测试 run_pretrain 完整流程"""

    def test_train_completes(
        self, tiny_yaml_config, sample_pretrain_data, saved_tmp_tokenizer, tmp_path
    ):
        """运行 2 步训练不报错"""
        output_dir = str(tmp_path / "pretrain_output")
        run_pretrain(
            config_path=tiny_yaml_config,
            data_path=sample_pretrain_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        assert Path(output_dir).exists()

    def test_output_files(
        self, tiny_yaml_config, sample_pretrain_data, saved_tmp_tokenizer, tmp_path
    ):
        """输出目录包含 config.json 和模型文件"""
        output_dir = str(tmp_path / "pretrain_output")
        run_pretrain(
            config_path=tiny_yaml_config,
            data_path=sample_pretrain_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        output_path = Path(output_dir)
        assert (output_path / "config.json").exists()
        # safetensors 或 bin 格式
        has_model = (
            (output_path / "model.safetensors").exists()
            or (output_path / "pytorch_model.bin").exists()
        )
        assert has_model

    def test_model_loadable(
        self, tiny_yaml_config, sample_pretrain_data, saved_tmp_tokenizer, tmp_path
    ):
        """保存的模型可用 from_pretrained 加载"""
        output_dir = str(tmp_path / "pretrain_output")
        run_pretrain(
            config_path=tiny_yaml_config,
            data_path=sample_pretrain_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        model = ClearMindForCausalLM.from_pretrained(output_dir)
        assert model is not None
        assert hasattr(model, "generate")


# ============================================================
# TestClearMindLoggingCallback
# ============================================================


class TestClearMindLoggingCallback:
    """测试 ClearMindLoggingCallback"""

    def test_instantiation(self):
        """可正常实例化"""
        callback = ClearMindLoggingCallback()
        assert callback is not None

    def test_is_trainer_callback(self):
        """是 TrainerCallback 子类"""
        callback = ClearMindLoggingCallback()
        assert isinstance(callback, TrainerCallback)


# ============================================================
# TestCreateSFTArgs
# ============================================================


class TestCreateSFTArgs:
    """测试 create_sft_args 函数"""

    def test_basic_fields(self, tmp_path):
        """基础字段正确透传"""
        config = {
            "per_device_train_batch_size": 4,
            "num_train_epochs": 2,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "logging_steps": 5,
            "bf16": False,
            "report_to": "none",
        }
        args = create_sft_args(config, str(tmp_path / "output"))
        assert args.per_device_train_batch_size == 4
        assert args.num_train_epochs == 2
        assert args.learning_rate == 2e-5

    def test_default_scheduler(self, tmp_path):
        """默认使用 cosine 调度器"""
        config = {"num_train_epochs": 1, "report_to": "none"}
        args = create_sft_args(config, str(tmp_path / "output"))
        assert args.lr_scheduler_type.value == "cosine"

    def test_default_eval_strategy(self, tmp_path):
        """默认按 epoch 评估"""
        config = {"num_train_epochs": 1, "report_to": "none"}
        args = create_sft_args(config, str(tmp_path / "output"))
        assert args.eval_strategy.value == "epoch"


# ============================================================
# TestRunSFT
# ============================================================


class TestRunSFT:
    """测试 run_sft 完整流程"""

    def test_sft_runs(
        self, tiny_yaml_config, sample_sft_data, saved_tmp_tokenizer, tmp_path
    ):
        """SFT 运行 2 步不报错"""
        output_dir = str(tmp_path / "sft_output")
        run_sft(
            config_path=tiny_yaml_config,
            data_path=sample_sft_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        assert Path(output_dir).exists()

    def test_sft_output_files(
        self, tiny_yaml_config, sample_sft_data, saved_tmp_tokenizer, tmp_path
    ):
        """SFT 输出包含 config.json 和模型文件"""
        output_dir = str(tmp_path / "sft_output")
        run_sft(
            config_path=tiny_yaml_config,
            data_path=sample_sft_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        output_path = Path(output_dir)
        assert (output_path / "config.json").exists()
        has_model = (
            (output_path / "model.safetensors").exists()
            or (output_path / "pytorch_model.bin").exists()
        )
        assert has_model

    def test_sft_model_loadable(
        self, tiny_yaml_config, sample_sft_data, saved_tmp_tokenizer, tmp_path
    ):
        """SFT 保存的模型可用 from_pretrained 加载"""
        output_dir = str(tmp_path / "sft_output")
        run_sft(
            config_path=tiny_yaml_config,
            data_path=sample_sft_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        model = ClearMindForCausalLM.from_pretrained(output_dir)
        assert model is not None

    def test_sft_from_pretrained(
        self, tiny_yaml_config, sample_pretrain_data, sample_sft_data,
        saved_tmp_tokenizer, tmp_path
    ):
        """SFT 可从预训练模型继续微调"""
        # 先做一步 pretrain
        pretrain_dir = str(tmp_path / "pretrain_output")
        run_pretrain(
            config_path=tiny_yaml_config,
            data_path=sample_pretrain_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=pretrain_dir,
            max_steps=1,
        )

        # 再做 SFT
        sft_dir = str(tmp_path / "sft_output")
        run_sft(
            config_path=tiny_yaml_config,
            data_path=sample_sft_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=sft_dir,
            pretrained_path=pretrain_dir,
            max_steps=1,
        )
        assert Path(sft_dir).exists()
        assert (Path(sft_dir) / "config.json").exists()


# ============================================================
# TestSFTLabelsMask
# ============================================================


class TestSFTLabelsMask:
    """验证 SFT labels mask 正确性"""

    def test_labels_have_mask(self, tmp_tokenizer, sample_sft_data):
        """SFT 数据的 labels 包含 -100 mask"""
        from data import load_sft_dataset

        result = load_sft_dataset(sample_sft_data, tmp_tokenizer, max_length=128)
        for sample in result["train"]:
            labels = sample["labels"]
            has_masked = any(l == -100 for l in labels)
            has_valid = any(l != -100 for l in labels)
            assert has_masked, "应有 -100 (prompt 部分)"
            assert has_valid, "应有有效 token (response 部分)"

    def test_labels_mask_at_start(self, tmp_tokenizer, sample_sft_data):
        """-100 出现在序列开头 (prompt 部分)，有效 token 在后面 (response)"""
        from data import load_sft_dataset

        result = load_sft_dataset(sample_sft_data, tmp_tokenizer, max_length=128)
        for sample in result["train"]:
            labels = sample["labels"]
            first_valid = next(i for i, l in enumerate(labels) if l != -100)
            # 前面全是 -100
            assert all(l == -100 for l in labels[:first_valid])


# ============================================================
# TestCreateDPOArgs
# ============================================================


class TestCreateDPOArgs:
    """测试 create_dpo_args 函数"""

    def test_basic_fields(self, tmp_path):
        """基础字段正确透传"""
        config = {
            "per_device_train_batch_size": 2,
            "num_train_epochs": 1,
            "learning_rate": 1e-5,
            "beta": 0.1,
            "logging_steps": 5,
            "bf16": False,
            "report_to": "none",
        }
        args = create_dpo_args(config, str(tmp_path / "output"))
        assert args.per_device_train_batch_size == 2
        assert args.learning_rate == 1e-5
        assert args.beta == 0.1

    def test_beta_passthrough(self, tmp_path):
        """beta 参数正确传递到 DPOConfig"""
        config = {"beta": 0.2, "report_to": "none"}
        args = create_dpo_args(config, str(tmp_path / "output"))
        assert args.beta == 0.2


# ============================================================
# TestRunDPO
# ============================================================


class TestRunDPO:
    """测试 run_dpo 完整流程"""

    def test_dpo_runs(
        self, tiny_yaml_config, sample_dpo_data, saved_tmp_tokenizer, tmp_path
    ):
        """DPO 运行 2 步不报错"""
        output_dir = str(tmp_path / "dpo_output")
        run_dpo(
            config_path=tiny_yaml_config,
            data_path=sample_dpo_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        assert Path(output_dir).exists()

    def test_dpo_output_files(
        self, tiny_yaml_config, sample_dpo_data, saved_tmp_tokenizer, tmp_path
    ):
        """DPO 输出包含 config.json 和模型文件"""
        output_dir = str(tmp_path / "dpo_output")
        run_dpo(
            config_path=tiny_yaml_config,
            data_path=sample_dpo_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        output_path = Path(output_dir)
        assert (output_path / "config.json").exists()
        has_model = (
            (output_path / "model.safetensors").exists()
            or (output_path / "pytorch_model.bin").exists()
        )
        assert has_model

    def test_dpo_model_loadable(
        self, tiny_yaml_config, sample_dpo_data, saved_tmp_tokenizer, tmp_path
    ):
        """DPO 保存的模型可用 from_pretrained 加载"""
        output_dir = str(tmp_path / "dpo_output")
        run_dpo(
            config_path=tiny_yaml_config,
            data_path=sample_dpo_data,
            tokenizer_path=saved_tmp_tokenizer,
            output_dir=output_dir,
            max_steps=2,
        )
        model = ClearMindForCausalLM.from_pretrained(output_dir)
        assert model is not None

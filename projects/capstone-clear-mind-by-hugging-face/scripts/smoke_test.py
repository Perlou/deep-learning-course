"""
smoke_test.py — 端到端冒烟测试
===============================

通过 subprocess 执行完整流程，验证从数据准备到推理的整个管线。

对应 from-scratch 版的 scripts/smoke_test.py：
  1. prepare_data.py --scale small
  2. train_tokenizer.py
  3. train.py --stage pretrain --max_steps 1
  4. 验证输出 + 推理

用法:
  python scripts/smoke_test.py
  python scripts/smoke_test.py --max_steps 2
  python scripts/smoke_test.py --clean
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
VENV_PYTHON = str(ROOT / "venv" / "bin" / "python")


def run_cmd(cmd: list[str], desc: str) -> None:
    """执行命令并检查返回码"""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n❌ 失败: {desc}")
        sys.exit(1)
    print(f"\n✅ 完成: {desc}")


def main():
    parser = argparse.ArgumentParser(description="ClearMind 端到端冒烟测试")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny.yaml",
        help="YAML 配置文件",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="outputs/smoke",
        help="工作目录",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1,
        help="预训练步数",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前清理工作目录",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    tok_dir = work_dir / "tokenizer"
    pretrain_dir = work_dir / "pretrain"

    # 清理
    if args.clean and work_dir.exists():
        print(f"清理工作目录: {work_dir}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("ClearMind 冒烟测试")
    print("=" * 60)
    print(f"  配置: {args.config}")
    print(f"  工作目录: {work_dir}")
    print(f"  预训练步数: {args.max_steps}")

    # Step 1: 准备数据
    run_cmd(
        [
            VENV_PYTHON,
            "scripts/prepare_data.py",
            "--scale", "small",
            "--data_dir", str(data_dir),
        ],
        "Step 1: 准备数据",
    )

    # Step 2: 训练 Tokenizer
    corpus_path = data_dir / "pretrain" / "pretrain_data.jsonl"
    run_cmd(
        [
            VENV_PYTHON,
            "scripts/train_tokenizer.py",
            "--config", args.config,
            "--corpus", str(corpus_path),
            "--output_dir", str(tok_dir),
        ],
        "Step 2: 训练 Tokenizer",
    )

    # Step 3: 预训练
    pretrain_data = data_dir / "pretrain" / "pretrain_data.jsonl"
    run_cmd(
        [
            VENV_PYTHON,
            "scripts/train.py",
            "--stage", "pretrain",
            "--config", args.config,
            "--data", str(pretrain_data),
            "--tokenizer", str(tok_dir),
            "--output_dir", str(pretrain_dir),
            "--max_steps", str(args.max_steps),
        ],
        "Step 3: 预训练",
    )

    # Step 4: 验证输出
    print(f"\n{'=' * 60}")
    print("  Step 4: 验证输出")
    print(f"{'=' * 60}\n")

    config_file = pretrain_dir / "config.json"
    if not config_file.exists():
        print("❌ config.json 不存在")
        sys.exit(1)
    print("✅ config.json 存在")

    has_model = (
        (pretrain_dir / "model.safetensors").exists()
        or (pretrain_dir / "pytorch_model.bin").exists()
    )
    if not has_model:
        print("❌ 模型文件不存在")
        sys.exit(1)
    print("✅ 模型文件存在")

    # Step 5: 加载模型并验证推理
    print(f"\n{'=' * 60}")
    print("  Step 5: 验证推理")
    print(f"{'=' * 60}\n")

    # 将 src/ 加入路径
    sys.path.insert(0, str(ROOT / "src"))
    from model import ClearMindForCausalLM
    from data.tokenizer import ClearMindTokenizer

    model = ClearMindForCausalLM.from_pretrained(str(pretrain_dir))
    model.eval()
    tokenizer = ClearMindTokenizer.load(str(tok_dir))

    inputs = tokenizer("你好", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  输入: '你好'")
    print(f"  输出: '{text}'")

    if len(text) == 0:
        print("❌ 推理输出为空")
        sys.exit(1)
    print("✅ 推理验证通过")

    # 总结
    print(f"\n{'=' * 60}")
    print("🎉 冒烟测试全部通过！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

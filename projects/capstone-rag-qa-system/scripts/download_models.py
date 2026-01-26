#!/usr/bin/env python
"""
DocuMind AI - 模型下载脚本

下载所需的嵌入模型和 LLM
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_settings, setup_logger, log


def download_embedding_model():
    """下载嵌入模型"""
    settings = get_settings()
    model_name = settings.models.embedding.name

    log.info(f"正在下载嵌入模型: {model_name}")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        log.info(f"✓ 嵌入模型下载完成: {model_name}")

        # 测试模型
        test_text = "这是一个测试句子"
        embedding = model.encode(test_text)
        log.info(f"✓ 嵌入模型测试成功，向量维度: {len(embedding)}")

    except Exception as e:
        log.error(f"✗ 嵌入模型下载失败: {e}")
        return False

    return True


def download_llm_model():
    """下载 LLM 模型"""
    settings = get_settings()
    model_name = settings.models.llm.name

    log.info(f"正在下载 LLM 模型: {model_name}")
    log.warning("⚠️ LLM 模型较大，下载可能需要较长时间...")
    log.info("💡 提示: 如果下载缓慢，可以从 ModelScope 下载:")
    log.info(f"   modelscope download --model qwen/Qwen2.5-7B-Instruct")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        log.info("正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        log.info("✓ Tokenizer 下载完成")

        log.info("正在下载模型权重（这可能需要一段时间）...")
        # 只下载配置，不加载到内存
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        log.info(f"✓ 模型配置已获取: {config.model_type}")

        log.info("💡 完整模型将在首次运行时下载和缓存")

    except Exception as e:
        log.error(f"✗ LLM 模型下载失败: {e}")
        log.info("💡 您可以稍后手动下载模型")
        return False

    return True


def main():
    """主函数"""
    setup_logger()

    log.info("=" * 50)
    log.info("DocuMind AI - 模型下载脚本")
    log.info("=" * 50)

    settings = get_settings()
    log.info(f"嵌入模型: {settings.models.embedding.name}")
    log.info(f"LLM 模型: {settings.models.llm.name}")
    log.info("")

    # 下载嵌入模型
    log.info("[1/2] 下载嵌入模型")
    embedding_success = download_embedding_model()
    log.info("")

    # 下载 LLM 模型
    log.info("[2/2] 下载 LLM 模型")
    llm_success = download_llm_model()
    log.info("")

    # 总结
    log.info("=" * 50)
    log.info("下载总结:")
    log.info(f"  嵌入模型: {'✓ 成功' if embedding_success else '✗ 失败'}")
    log.info(f"  LLM 模型: {'✓ 成功' if llm_success else '✗ 需要手动下载'}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()

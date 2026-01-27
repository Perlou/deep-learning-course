"""
DocuMind AI - LLM 推理引擎模块

实现大语言模型加载和推理功能
支持 Qwen2.5-7B-Instruct 等模型
"""

from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, Generator, List, Optional

import torch

from src.utils import get_settings, log


@dataclass
class GenerationConfig:
    """生成参数配置"""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class LLMEngine:
    """
    LLM 推理引擎

    负责模型加载和文本生成
    支持普通生成和流式生成
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        初始化 LLM 引擎

        Args:
            model_name: 模型名称或路径
            device: 设备 (auto/cuda/cpu)
            quantization: 量化方式 (none/int4/int8)
            use_mock: 是否使用 Mock 模式（用于测试）
        """
        settings = get_settings()
        llm_config = settings.models.llm

        self.model_name = model_name or llm_config.name
        self.quantization = quantization or llm_config.quantization
        self.use_mock = use_mock

        # 设备选择
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 生成参数
        self.default_config = GenerationConfig(
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            repetition_penalty=llm_config.repetition_penalty,
        )

        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        log.info(
            f"LLMEngine 初始化: model={self.model_name}, "
            f"device={self.device}, quantization={self.quantization}, "
            f"mock={self.use_mock}"
        )

    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            是否加载成功
        """
        if self._is_loaded:
            log.debug("模型已加载，跳过")
            return True

        if self.use_mock:
            log.info("使用 Mock 模式，跳过模型加载")
            self._is_loaded = True
            return True

        try:
            log.info(f"开始加载模型: {self.model_name}")

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # 构建加载参数
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # 设备和数据类型配置
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16

                # 量化配置
                if self.quantization == "int4":
                    try:
                        from transformers import BitsAndBytesConfig

                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        log.info("使用 INT4 量化")
                    except ImportError:
                        log.warning("bitsandbytes 未安装，跳过量化")

                elif self.quantization == "int8":
                    try:
                        from transformers import BitsAndBytesConfig

                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                        log.info("使用 INT8 量化")
                    except ImportError:
                        log.warning("bitsandbytes 未安装，跳过量化")
            else:
                # CPU 模式
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.float32
                log.info("使用 CPU 模式（推理速度较慢）")

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self._is_loaded = True
            log.info(f"模型加载完成: {self.model_name}")
            return True

        except Exception as e:
            log.error(f"模型加载失败: {e}")
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        """是否已加载模型"""
        return self._is_loaded

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        生成文本（非流式）

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: top-p 采样参数
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        if not self._is_loaded:
            if not self.load_model():
                return "错误：模型未加载"

        # Mock 模式
        if self.use_mock:
            return self._mock_generate(prompt)

        # 合并参数
        max_tokens = max_tokens or self.default_config.max_tokens
        temperature = temperature or self.default_config.temperature
        top_p = top_p or self.default_config.top_p

        try:
            # 构建输入
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=self.default_config.do_sample,
                    repetition_penalty=self.default_config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 解码
            generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            log.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}"

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: top-p 采样参数
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        if not self._is_loaded:
            if not self.load_model():
                yield "错误：模型未加载"
                return

        # Mock 模式
        if self.use_mock:
            yield from self._mock_stream_generate(prompt)
            return

        # 合并参数
        max_tokens = max_tokens or self.default_config.max_tokens
        temperature = temperature or self.default_config.temperature
        top_p = top_p or self.default_config.top_p

        try:
            from transformers import TextIteratorStreamer

            # 构建输入
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            # 创建 streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # 生成参数
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": self.default_config.do_sample,
                "repetition_penalty": self.default_config.repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

            # 在线程中生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式输出
            for token in streamer:
                if token:
                    yield token

            thread.join()

        except Exception as e:
            log.error(f"流式生成失败: {e}")
            yield f"生成失败: {str(e)}"

    def _mock_generate(self, prompt: str) -> str:
        """Mock 模式生成（用于测试）"""
        return f"[Mock 响应] 收到问题：{prompt[:100]}..."

    def _mock_stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """Mock 模式流式生成（用于测试）"""
        import time

        response = f"[Mock 响应] 收到问题：{prompt[:50]}...\n\n这是一个模拟的流式响应，用于测试目的。"
        words = response.split()

        for word in words:
            yield word + " "
            time.sleep(0.05)

    def unload_model(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        log.info("模型已卸载")


# ============================================
# 全局单例
# ============================================
_llm_engine: Optional[LLMEngine] = None


def get_llm_engine(use_mock: bool = False) -> LLMEngine:
    """
    获取 LLM 引擎单例

    Args:
        use_mock: 是否使用 Mock 模式

    Returns:
        LLMEngine 实例
    """
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine(use_mock=use_mock)
    return _llm_engine


def init_llm_engine(use_mock: bool = False) -> bool:
    """
    初始化并预加载 LLM 引擎

    Args:
        use_mock: 是否使用 Mock 模式

    Returns:
        是否加载成功
    """
    engine = get_llm_engine(use_mock=use_mock)
    return engine.load_model()

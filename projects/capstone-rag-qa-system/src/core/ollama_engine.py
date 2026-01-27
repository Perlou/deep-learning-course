"""
DocuMind AI - Ollama LLM 引擎模块

使用 Ollama API 进行本地 LLM 推理
支持 qwen2.5:7b 等模型
"""

import json
import os
from typing import Generator, Optional

import httpx

from src.utils import get_settings, log


class OllamaEngine:
    """
    Ollama LLM 引擎

    通过 Ollama API 调用本地运行的 LLM 模型
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        初始化 Ollama 引擎

        Args:
            model: 模型名称，如 "qwen2.5:7b"
            base_url: Ollama API 地址，默认 http://localhost:11434
            timeout: 请求超时时间（秒）
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout

        settings = get_settings()
        llm_config = settings.models.llm

        # 生成参数
        self.default_options = {
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "num_predict": llm_config.max_tokens,
            "repeat_penalty": llm_config.repetition_penalty,
        }

        self._is_available = None
        log.info(f"OllamaEngine 初始化: model={self.model}, base_url={self.base_url}")

    def check_availability(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    self._is_available = self.model in models or any(
                        self.model.split(":")[0] in m for m in models
                    )
                    if self._is_available:
                        log.info(f"Ollama 模型 {self.model} 可用")
                    else:
                        log.warning(
                            f"Ollama 模型 {self.model} 未找到，可用模型: {models}"
                        )
                    return self._is_available
        except Exception as e:
            log.error(f"Ollama 服务连接失败: {e}")
            self._is_available = False
        return False

    @property
    def is_available(self) -> bool:
        """是否可用"""
        if self._is_available is None:
            self.check_availability()
        return self._is_available

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        生成文本（非流式）

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成的文本
        """
        options = self.default_options.copy()
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")

        except httpx.TimeoutException:
            log.error("Ollama 请求超时")
            return "错误：LLM 请求超时，请重试"
        except Exception as e:
            log.error(f"Ollama 生成失败: {e}")
            return f"错误：{str(e)}"

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成文本

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Yields:
            生成的文本片段
        """
        options = self.default_options.copy()
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException:
            log.error("Ollama 流式请求超时")
            yield "错误：LLM 请求超时"
        except Exception as e:
            log.error(f"Ollama 流式生成失败: {e}")
            yield f"错误：{str(e)}"

    def chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        对话式生成（使用 Ollama chat API）

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成的回复
        """
        options = self.default_options.copy()
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")

        except Exception as e:
            log.error(f"Ollama chat 失败: {e}")
            return f"错误：{str(e)}"


# 全局单例
_ollama_engine: Optional[OllamaEngine] = None


def get_ollama_engine() -> OllamaEngine:
    """获取 Ollama 引擎单例"""
    global _ollama_engine
    if _ollama_engine is None:
        _ollama_engine = OllamaEngine()
    return _ollama_engine

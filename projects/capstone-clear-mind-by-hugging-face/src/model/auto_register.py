"""
AutoClass 注册 — 让 ClearMind 模型支持 AutoConfig / AutoModelForCausalLM
"""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_clearmind import ClearMindConfig
from .modeling_clearmind import ClearMindForCausalLM

AutoConfig.register("clearmind", ClearMindConfig)
AutoModelForCausalLM.register(ClearMindConfig, ClearMindForCausalLM)

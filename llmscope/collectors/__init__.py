"""采集器注册表。

注册顺序 = 端口指纹判定的优先级:
SGLang 必须在 vLLM 之前(两者都暴露 /v1/models,
SGLang 的 /get_model_info 是区分二者的关键特征);
vLLM 又必须在 Ollama 之前(Ollama 同样实现了 /v1/models)。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseEngineCollector  # noqa: F401

COLLECTORS: list[type[BaseEngineCollector]] = []


def register(cls: type[BaseEngineCollector]) -> type[BaseEngineCollector]:
    """类装饰器:把采集器加入注册表。"""
    COLLECTORS.append(cls)
    return cls


def fingerprint_port(host: str, port: int):
    """尝试所有已注册采集器,返回 (采集器类, 指纹快照) 或 None。"""
    for cls in COLLECTORS:
        try:
            snap = cls.fingerprint(host, port)
        except Exception:
            continue
        if snap is not None:
            return cls, snap
    return None


# 导入子模块触发注册(顺序即优先级,勿随意调整)
from .sglang import SglangCollector  # noqa: E402,F401
from .vllm import VllmCollector      # noqa: E402,F401
from .ollama import OllamaCollector  # noqa: E402,F401

COLLECTOR_BY_NAME: dict[str, type[BaseEngineCollector]] = {
    cls.engine_name: cls for cls in COLLECTORS
}

"""vLLM 采集器:OpenAI 兼容 API + Prometheus /metrics。

数据来源:
    /version     → 引擎版本(旧版本可能没有,缺失时显示 unknown)
    /v1/models   → 当前服务化的模型
    /metrics     → 运行队列 / KV 缓存 / token 计数器(差分得吞吐)
"""
from __future__ import annotations

import time
from typing import Optional

from .. import models
from ..config import SETTINGS
from ..models import EngineSnapshot
from . import register
from .base import BaseEngineCollector, FP_TIMEOUT, get_json, http_get
from .prometheus import as_int, max_metric, parse_prometheus, sum_metric


@register
class VllmCollector(BaseEngineCollector):
    engine_name = models.VLLM
    default_ports = (8000,)
    process_keywords = ("vllm",)

    # 累计型 token 计数器;不同大版本若改名,按顺序向后加备选名即可
    GEN_COUNTER = "vllm:generation_tokens_total"
    PROMPT_COUNTER = "vllm:prompt_tokens_total"

    def __init__(self) -> None:
        # counter 差分状态:{(host, port, 组名): (上次时间, 上次计数值)}
        self._last: dict[tuple, tuple[float, Optional[float]]] = {}

    # ------------------------------------------------------------------
    # 指纹:区分"恰好开着 /v1/models 的其它服务"。
    # 注意 llama.cpp(含 Ollama 内嵌引擎)的 /v1/models 同时返回
    # OpenAI 与 Ollama 两种形状,单看 /v1/models 会误判;
    # 因此额外要求 vLLM 独有信号:/version 端点 或 /metrics 中的 vllm: 指标。
    # ------------------------------------------------------------------
    @classmethod
    def fingerprint(cls, host: str, port: int) -> Optional[EngineSnapshot]:
        base = cls._base(host, port)
        # SGLang 特征路径命中 → 不是 vLLM
        info = get_json(f"{base}/get_model_info", timeout=FP_TIMEOUT)
        if isinstance(info, dict) and "model_path" in info:
            return None
        # Ollama 特征路径命中 → 不是 vLLM
        ver = get_json(f"{base}/api/version", timeout=FP_TIMEOUT)
        if isinstance(ver, dict) and "version" in ver:
            return None
        version = None
        v = get_json(f"{base}/version", timeout=FP_TIMEOUT)
        if isinstance(v, dict) and "version" in v:
            version = str(v["version"])
        data = get_json(f"{base}/v1/models", timeout=FP_TIMEOUT)
        if not (isinstance(data, dict) and data.get("object") == "list" and data.get("data")):
            return None
        # 决定性信号:有 /version,或 /metrics 里有 vllm: 前缀的指标
        if version is None:
            try:
                r = http_get(f"{base}/metrics", timeout=FP_TIMEOUT)
                if r.status_code != 200 or "vllm:" not in r.text:
                    return None
            except Exception:
                return None
        return EngineSnapshot(
            engine=cls.engine_name, host=host, port=port,
            status="online", version=version,
        )

    # ------------------------------------------------------------------
    def _rate(self, key: tuple, value: Optional[float], now: float) -> Optional[float]:
        """对累计 counter 做差分求速率;计数器回退(服务重启)时丢弃一次。"""
        prev = self._last.get(key)
        self._last[key] = (now, value)
        if value is None or prev is None or prev[1] is None:
            return None
        dt = now - prev[0]
        dv = value - prev[1]
        if dt <= 0 or dv < 0:
            return None
        return dv / dt

    def collect(
        self, host: str, port: int, prev: Optional[EngineSnapshot] = None
    ) -> EngineSnapshot:
        base = self._base(host, port)
        snap = EngineSnapshot(engine=self.engine_name, host=host, port=port, status="online")
        if prev:
            snap.version, snap.pids = prev.version, prev.pids

        v = get_json(f"{base}/version")
        if isinstance(v, dict) and "version" in v:
            snap.version = str(v["version"])

        data = get_json(f"{base}/v1/models")
        if isinstance(data, dict) and data.get("data"):
            snap.model = str(data["data"][0].get("id") or "") or None

        t0 = time.perf_counter()
        metrics_text = None
        try:
            r = http_get(f"{base}/metrics")
            if r.status_code == 200:
                metrics_text = r.text
        except Exception:
            pass
        snap.latency_ms = (time.perf_counter() - t0) * 1000

        if metrics_text:
            # 详情页要展示 /metrics 原文;超长截断防止内存膨胀
            snap.extra["metrics_text"] = metrics_text[:100_000]
            parsed = parse_prometheus(metrics_text)
            snap.running_requests = as_int(sum_metric(parsed, "vllm:num_requests_running"))
            snap.queue_depth = as_int(sum_metric(parsed, "vllm:num_requests_waiting"))
            snap.preempted_requests = as_int(
                sum_metric(parsed, "vllm:num_requests_preempted")
            )
            # 新旧版本缓存占用指标名不同,依次尝试
            snap.kv_cache_usage = max_metric(parsed, "vllm:gpu_cache_usage_perc")
            if snap.kv_cache_usage is None:
                snap.kv_cache_usage = max_metric(parsed, "vllm:kv_cache_usage_perc")
            now = time.perf_counter()
            snap.gen_tokens_per_sec = self._rate(
                (host, port, "gen"), sum_metric(parsed, self.GEN_COUNTER), now
            )
            snap.prompt_tokens_per_sec = self._rate(
                (host, port, "prompt"), sum_metric(parsed, self.PROMPT_COUNTER), now
            )
        else:
            snap.error = "/metrics 不可用(若启用了 --api-key 需放行本机访问)"
        return snap

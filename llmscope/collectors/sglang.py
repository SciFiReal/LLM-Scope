"""SGLang 采集器:原生 HTTP API + Prometheus /metrics。

数据来源:
    /get_model_info   → 模型路径(同时是指纹特征)
    /get_server_info  → 引擎版本(旧版本可能没有)
    /metrics          → 运行队列 / 令牌池占用 / 吞吐 gauge
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .. import models
from ..models import EngineSnapshot
from . import register
from .base import BaseEngineCollector, FP_TIMEOUT, get_json, http_get
from .prometheus import as_int, max_metric, parse_prometheus, sum_metric


@register
class SglangCollector(BaseEngineCollector):
    engine_name = models.SGLANG
    default_ports = (30000,)
    process_keywords = ("sglang",)

    # 吞吐:SGLang 直接暴露 gauge;旧版本只有累计 counter 时走差分兜底
    GEN_GAUGE = "sglang:gen_throughput"
    PROMPT_GAUGE = "sglang:prompt_throughput"
    GEN_COUNTERS = ("sglang:generation_tokens_total", "sglang:num_generated_tokens")

    def __init__(self) -> None:
        self._last: dict[tuple, tuple[float, Optional[float]]] = {}

    # ------------------------------------------------------------------
    @classmethod
    def fingerprint(cls, host: str, port: int) -> Optional[EngineSnapshot]:
        base = cls._base(host, port)
        info = get_json(f"{base}/get_model_info", timeout=FP_TIMEOUT)
        if isinstance(info, dict) and "model_path" in info:
            return EngineSnapshot(
                engine=cls.engine_name, host=host, port=port, status="online"
            )
        return None

    # ------------------------------------------------------------------
    def _rate(self, key: tuple, value: Optional[float], now: float) -> Optional[float]:
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

        info = get_json(f"{base}/get_server_info")
        if isinstance(info, dict):
            ver = info.get("version") or info.get("sglang_version")
            if ver:
                snap.version = str(ver)

        info = get_json(f"{base}/get_model_info")
        if isinstance(info, dict) and info.get("model_path"):
            snap.model = os.path.basename(str(info["model_path"]))

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
            snap.running_requests = as_int(sum_metric(parsed, "sglang:num_running_reqs"))
            snap.queue_depth = as_int(sum_metric(parsed, "sglang:num_queue_reqs"))
            snap.preempted_requests = as_int(
                sum_metric(parsed, "sglang:num_retracted_reqs")
            )
            # 令牌池占用即 KV 缓存占用的等价物
            snap.kv_cache_usage = max_metric(parsed, "sglang:token_usage")

            # 吞吐优先取 gauge;没有 gauge 的旧版本用 counter 差分兜底
            snap.gen_tokens_per_sec = max_metric(parsed, self.GEN_GAUGE)
            snap.prompt_tokens_per_sec = max_metric(parsed, self.PROMPT_GAUGE)
            now = time.perf_counter()
            if snap.gen_tokens_per_sec is None:
                counter = next(
                    (sum_metric(parsed, n) for n in self.GEN_COUNTERS
                     if sum_metric(parsed, n) is not None),
                    None,
                )
                snap.gen_tokens_per_sec = self._rate((host, port, "gen"), counter, now)
        else:
            snap.error = "/metrics 不可用"
        return snap

"""Ollama 采集器:原生 API + 后台流式探针估算实时 token/s。

数据来源:
    /api/version → 版本(同时是指纹特征)
    /api/tags    → 已安装模型列表
    /api/ps      → 已加载模型及其显存占用(size_vram)
    /api/generate(流式,探针)→ eval_count / eval_duration 得生成速度

探针说明:只对"已经加载"的模型发送一个极小的生成请求
(默认 8 token、间隔 30s,可在 TUI 按 o 开关),不会把冷模型拉进显存。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

import requests

from .. import models
from ..config import SETTINGS
from . import register
from .base import BaseEngineCollector, FP_TIMEOUT, get_json
from ..models import EngineSnapshot, ModelSlot


@register
class OllamaCollector(BaseEngineCollector):
    engine_name = models.OLLAMA
    default_ports = (11434,)
    process_keywords = ("ollama",)

    def __init__(self) -> None:
        self._last_probe_t = 0.0
        self._probe_lock = threading.Lock()

    # ------------------------------------------------------------------
    @classmethod
    def fingerprint(cls, host: str, port: int) -> Optional[EngineSnapshot]:
        base = cls._base(host, port)
        ver = get_json(f"{base}/api/version", timeout=FP_TIMEOUT)
        if not (isinstance(ver, dict) and "version" in ver):
            return None
        # Ollama 桌面版的 WebView GUI 也会代理 /api/version、/api/tags,
        # 但未覆盖的路径(如 /api/ps)会回落成 HTML 页面;
        # 真正的 API 服务对 /api/ps 永远返回 JSON(空闲时也是 {"models":[]})。
        # 用这一点把 GUI 端口排除,避免出现重复的"伪实例"。
        ps = get_json(f"{base}/api/ps", timeout=FP_TIMEOUT)
        if not (isinstance(ps, dict) and "models" in ps):
            return None
        return EngineSnapshot(
            engine=cls.engine_name, host=host, port=port,
            status="online", version=str(ver["version"]),
        )

    # ------------------------------------------------------------------
    def collect(
        self, host: str, port: int, prev: Optional[EngineSnapshot] = None
    ) -> EngineSnapshot:
        base = self._base(host, port)
        snap = EngineSnapshot(engine=self.engine_name, host=host, port=port, status="online")
        if prev:
            snap.version, snap.pids = prev.version, prev.pids

        ver = get_json(f"{base}/api/version")
        if isinstance(ver, dict) and "version" in ver:
            snap.version = str(ver["version"])

        tags = get_json(f"{base}/api/tags")
        if isinstance(tags, dict):
            snap.extra["installed"] = [
                m.get("name", "") for m in tags.get("models", [])
            ]

        ps = get_json(f"{base}/api/ps")
        loaded: list[ModelSlot] = []
        if isinstance(ps, dict):
            for m in ps.get("models", []):
                loaded.append(
                    ModelSlot(
                        name=str(m.get("name", "")),
                        vram_bytes=m.get("size_vram"),
                        size_bytes=m.get("size"),
                        quantization=(m.get("details") or {}).get("quantization_level"),
                    )
                )
        snap.models = loaded
        if loaded:
            snap.model = loaded[0].name
            snap.vram_bytes = sum(m.vram_bytes or 0 for m in loaded) or None
            # 权重没全进显存的比例 = 1 - vram/size(取最严重的那个模型)
            offloads = [
                max(0.0, 1.0 - (m.vram_bytes or 0) / m.size_bytes)
                for m in loaded
                if m.size_bytes and m.vram_bytes is not None
            ]
            snap.cpu_offload_ratio = max(offloads) if offloads else 0.0
        elif ps is None and prev:
            # 服务短暂无响应时沿用上次的模型信息,避免 UI 闪烁
            snap.models, snap.model = prev.models, prev.model
            snap.vram_bytes = prev.vram_bytes
            snap.cpu_offload_ratio = prev.cpu_offload_ratio
            snap.error = "/api/ps 暂时不可用"

        self._maybe_probe(base, snap, prev)
        return snap

    # ------------------------------------------------------------------
    def _maybe_probe(
        self, base: str, snap: EngineSnapshot, prev: Optional[EngineSnapshot]
    ) -> None:
        """按开关 / 间隔调度测速探针;未探针时沿用上次结果,UI 不闪烁。"""
        if not snap.models:
            return
        if not SETTINGS.ollama_probe_enabled:
            snap.extra["probe"] = "disabled"
            return

        now = time.perf_counter()
        due = now - self._last_probe_t >= SETTINGS.ollama_probe_interval
        if due and self._probe_lock.acquire(blocking=False):
            self._last_probe_t = now
            try:
                gen, prefill = self._probe(base, snap.model or "")
                snap.gen_tokens_per_sec = gen
                snap.prompt_tokens_per_sec = prefill
                snap.extra["probe"] = "done"
            except Exception as exc:
                snap.extra["probe"] = f"failed: {exc}"
            finally:
                self._probe_lock.release()
        else:
            snap.extra["probe"] = "cached"
            if prev:
                snap.gen_tokens_per_sec = prev.gen_tokens_per_sec
                snap.prompt_tokens_per_sec = prev.prompt_tokens_per_sec

    def _probe(self, base: str, model: str) -> tuple[Optional[float], Optional[float]]:
        """发送一次极小流式请求,返回 (生成 tok/s, 预填 tok/s)。"""
        payload = {
            "model": model,
            "prompt": "用一句话介绍大海。",
            "stream": True,
            "options": {
                "num_predict": SETTINGS.ollama_probe_num_predict,
                "temperature": 0,
            },
        }
        gen_chunks = 0
        final = None
        t0 = time.perf_counter()
        with requests.post(
            f"{base}/api/generate",
            json=payload,
            stream=True,
            timeout=(SETTINGS.http_timeout, SETTINGS.ollama_probe_timeout),
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("done"):
                    final = obj
                    break
                gen_chunks += 1
        wall = time.perf_counter() - t0

        gen_tps = prefill_tps = None
        if final:
            eval_dur, eval_cnt = final.get("eval_duration"), final.get("eval_count")
            if eval_dur and eval_cnt:
                gen_tps = eval_cnt / (eval_dur / 1e9)
            p_dur, p_cnt = final.get("prompt_eval_duration"), final.get("prompt_eval_count")
            if p_dur and p_cnt:
                prefill_tps = p_cnt / (p_dur / 1e9)
        # 兜底:老版本没有统计字段时按流式分块数估算
        if gen_tps is None and gen_chunks and wall > 0:
            gen_tps = gen_chunks / wall
        return gen_tps, prefill_tps

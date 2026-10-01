"""引擎管理器:发现 → 采集器实例 → 并行采集。

UI 层只与 EngineManager 对话:
    rediscover()  定期调用,维护在线引擎列表(带容灾计数)
    collect_all() 线程池并行采集,返回本轮 EngineSnapshot 列表
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from .collectors import COLLECTOR_BY_NAME
from .discovery import discover
from .models import EngineSnapshot


@dataclass
class _Entry:
    """一个在线引擎端点对应的采集器实例与其最近快照。"""

    collector: object
    snapshot: EngineSnapshot
    missed: int = 0  # 连续未发现的次数(容灾)


class EngineManager:
    """维护"在线引擎列表",并提供并行采集。

    容灾策略:服务短暂无响应(如正在加载权重)不会立即从列表消失,
    连续 GRACE 轮发现不到才移除,避免 UI 闪烁。
    """

    GRACE = 2

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int], _Entry] = {}
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="collect")

    # ---------- 发现 ----------
    def rediscover(self) -> None:
        found = {(ep.engine, ep.port): ep for ep in discover()}
        for key, ep in found.items():
            if key in self._entries:
                self._entries[key].missed = 0
                continue
            cls = COLLECTOR_BY_NAME.get(ep.engine)
            if cls is None:
                continue
            initial = EngineSnapshot(
                engine=ep.engine, host=ep.host, port=ep.port, pids=ep.pids
            )
            try:  # 再跑一次指纹,把版本号带进初始快照
                probe = cls.fingerprint(ep.host, ep.port)
                if probe:
                    initial.version = probe.version
            except Exception:
                pass
            self._entries[key] = _Entry(collector=cls(), snapshot=initial)

        for key in list(self._entries):
            if key not in found:
                entry = self._entries[key]
                entry.missed += 1
                if entry.missed > self.GRACE:
                    del self._entries[key]

    # ---------- 采集 ----------
    def collect_all(self) -> list[EngineSnapshot]:
        if not self._entries:
            return []
        futures = [self._pool.submit(self._collect_one, key) for key in self._entries]
        return [f.result() for f in futures]

    def _collect_one(self, key: tuple[str, int]) -> EngineSnapshot:
        entry = self._entries[key]
        host = entry.snapshot.host
        try:
            snap = entry.collector.collect(host, key[1], prev=entry.snapshot)
            snap.pids = snap.pids or entry.snapshot.pids
        except Exception as exc:
            old = entry.snapshot
            snap = EngineSnapshot(
                engine=key[0],
                host=host,
                port=key[1],
                status="unreachable",
                version=old.version,
                model=old.model,
                vram_bytes=old.vram_bytes,
                pids=old.pids,
                error=f"{type(exc).__name__}: {exc}",
            )
        entry.snapshot = snap
        return snap

    # ---------- 生命周期 ----------
    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def entries(self) -> dict[tuple[str, int], _Entry]:
        return dict(self._entries)

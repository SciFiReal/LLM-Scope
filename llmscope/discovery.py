"""引擎自动发现:默认端口扫描 + 进程识别双通道。

通道一(端口):直接探测各引擎默认端口 —— 覆盖 WSL / 容器等
    本机进程表看不到的场景;
通道二(进程):按进程名 / 命令行关键字匹配引擎进程,再取其
    监听的 TCP 端口 —— 覆盖自定义端口场景。

所有候选端口统一经指纹端点确认,避免把恰好占用同端口的
其它服务误判为推理引擎。
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

import psutil

from .collectors import COLLECTORS, fingerprint_port
from .config import SETTINGS


@dataclass(frozen=True)
class DiscoveredEndpoint:
    """一个确认在线的引擎端点。"""

    engine: str
    host: str
    port: int
    pids: frozenset = field(default_factory=frozenset)


def _scan_processes():
    """返回 (pid→监听端口集合, 引擎名→pid集合)。

    进程连接信息在部分系统需要提权,拿不到时返回空集合,
    发现已退化成"仅默认端口扫描",功能不受影响。
    """
    engine_pids: dict[str, set[int]] = defaultdict(set)
    pid_ports: dict[int, set[int]] = defaultdict(set)
    me = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        info = proc.info
        pid = info.get("pid")
        if not pid or pid == me:
            continue
        cmdline = info.get("cmdline") or []
        text = " ".join([info.get("name") or "", *cmdline]).lower()
        for cls in COLLECTORS:
            if any(k in text for k in cls.process_keywords):
                engine_pids[cls.engine_name].add(pid)

    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.pid and conn.laddr:
                pid_ports[conn.pid].add(conn.laddr.port)
    except psutil.Error:
        pass
    return pid_ports, engine_pids


def discover() -> list[DiscoveredEndpoint]:
    """执行一轮发现,返回所有指纹确认在线的引擎端点。"""
    pid_ports, engine_pids = _scan_processes()

    # 1) 收集候选端口:默认端口 + 引擎进程监听的端口(限量,防误扫)
    candidates: set[int] = set()
    for cls in COLLECTORS:
        candidates.update(cls.default_ports)
    process_ports: set[int] = set()
    for pids in engine_pids.values():
        for pid in pids:
            process_ports.update(pid_ports.get(pid, ()))
    candidates.update(sorted(process_ports)[: SETTINGS.max_process_ports])

    # 2) 逐端口指纹确认(按注册顺序,首个命中者胜出)
    results: list[DiscoveredEndpoint] = []
    seen: set[tuple[str, int]] = set()
    for port in sorted(candidates):
        hit = fingerprint_port("127.0.0.1", port)
        if hit is None:
            continue
        cls, _snap = hit
        key = (cls.engine_name, port)
        if key in seen:
            continue
        seen.add(key)
        # 把"监听该端口且命令行匹配本引擎"的 PID 归属给引擎,
        # 之后硬件层的 GPU 进程显存会按 PID 归账
        pids = {
            pid
            for pid in engine_pids.get(cls.engine_name, set())
            if port in pid_ports.get(pid, ())
        }
        results.append(
            DiscoveredEndpoint(cls.engine_name, "127.0.0.1", port, frozenset(pids))
        )
    return results

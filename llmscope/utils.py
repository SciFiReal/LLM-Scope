"""格式化小工具:字节数、速率、百分比、GPU 短名等统一显示。"""
from __future__ import annotations

import re
from typing import Optional

_BRAND_RE = re.compile(r"^(NVIDIA|AMD|Intel|INTEL)\s+")
_PREFIX_RE = re.compile(r"^(GeForce|Quadro|Tesla|Radeon|Arc|GRID|Data Center)\s+", re.IGNORECASE)
_TAIL_GPU_RE = re.compile(r"\s+GPU$", re.IGNORECASE)
_MEM_RE = re.compile(r"-(\d+GB)\b", re.IGNORECASE)


def short_gpu_name(full: str) -> str:
    """把 NVML 的完整显卡名压缩成用户关心的型号简称。

    "NVIDIA GeForce RTX 3050 Laptop GPU" → "RTX 3050 Laptop"
    "NVIDIA A100-SXM4-40GB"             → "A100 SXM4 40GB"
    完整名称仍可在面板副标题里查看。
    """
    name = full.strip()
    name = _BRAND_RE.sub("", name)
    name = _PREFIX_RE.sub("", name)
    name = _TAIL_GPU_RE.sub("", name)
    name = _MEM_RE.sub(lambda m: " " + m.group(1).upper(), name)
    name = name.replace("-", " ")
    return name.strip() or full


def human_bytes(n: Optional[float]) -> str:
    """字节数 → 人类可读(1024 进制)。None → '—'。"""
    if n is None:
        return "—"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "—"


def fmt_num(v: Optional[float], digits: int = 1) -> str:
    """数字 → 定点字符串;None → '—'。"""
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def fmt_rate(v: Optional[float], unit: str = "tok/s", digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f} {unit}"


def fmt_pct(ratio: Optional[float], digits: int = 0) -> str:
    """0~1 比例 → 百分比字符串;None → '—'。"""
    if ratio is None:
        return "—"
    return f"{ratio * 100:.{digits}f}%"


def trunc(s: Optional[str], width: int = 32) -> str:
    """超长字符串截断加省略号;空 → '—'。"""
    if not s:
        return "—"
    return s if len(s) <= width else s[: width - 1] + "…"

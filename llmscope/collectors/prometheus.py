"""极简 Prometheus 文本格式解析器。

vLLM / SGLang 的 /metrics 都输出 Prometheus 文本格式;
为保持"轻量、零配置",这里不引入 prometheus_client 依赖,
只解析本工具需要的 gauge / counter 数值(跨 label 求和 / 取最大)。
"""
from __future__ import annotations

import math
import re
from typing import Optional

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[^\s]+)"
    r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"')

# {metric_name: [(labels_dict, float_value), ...]}
Parsed = dict[str, list[tuple[dict, float]]]


def parse_prometheus(text: str) -> Parsed:
    """解析 Prometheus 文本暴露格式;非法行、NaN 值一律跳过。"""
    out: Parsed = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if math.isnan(value):
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def _series(parsed: Parsed, name: str) -> list[float]:
    return [v for _, v in parsed.get(name, ())]


def sum_metric(parsed: Parsed, name: str) -> Optional[float]:
    """同名指标(跨 label,如多卡/多模型)求和;缺失返回 None。"""
    vals = _series(parsed, name)
    return sum(vals) if vals else None


def max_metric(parsed: Parsed, name: str) -> Optional[float]:
    """同名指标取最大(如多副本时取最紧张的那个)。"""
    vals = _series(parsed, name)
    return max(vals) if vals else None


def as_int(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(round(v))

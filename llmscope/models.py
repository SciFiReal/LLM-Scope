"""数据模型:各采集器产出的统一快照结构。

所有模块只依赖这里的 dataclass 交换数据,彼此解耦;
新增引擎时复用 EngineSnapshot 即可,UI / 诊断层零改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# 引擎标识(字符串常量而非 Enum,便于第三方扩展注册新引擎)
OLLAMA = "ollama"
VLLM = "vllm"
SGLANG = "sglang"


@dataclass
class ModelSlot:
    """一个已加载到显存的模型实例。"""

    name: str
    vram_bytes: Optional[int] = None      # 当前显存占用
    size_bytes: Optional[int] = None      # 模型完整体积(权重)
    quantization: Optional[str] = None    # 量化格式,如 Q4_K_M
    extra: dict = field(default_factory=dict)


@dataclass
class EngineSnapshot:
    """单个引擎某次采集的完整快照;字段按引擎能力可选,拿不到就留 None。"""

    engine: str
    host: str = "127.0.0.1"
    port: int = 0
    status: str = "online"                          # online / unreachable
    version: Optional[str] = None
    model: Optional[str] = None                     # 主模型(当前活跃)
    models: list[ModelSlot] = field(default_factory=list)
    gen_tokens_per_sec: Optional[float] = None      # 生成吞吐(全局聚合)
    prompt_tokens_per_sec: Optional[float] = None   # 预填(prefill)吞吐
    kv_cache_usage: Optional[float] = None          # 0~1,KV 缓存/令牌池占用
    queue_depth: Optional[int] = None               # 等待队列长度
    running_requests: Optional[int] = None          # 正在处理的请求数
    preempted_requests: Optional[int] = None        # 被抢占的请求数(vLLM)
    vram_bytes: Optional[int] = None                # 引擎自身显存占用合计
    cpu_offload_ratio: Optional[float] = None       # 权重在内存中的比例(Ollama)
    latency_ms: Optional[float] = None              # /metrics 请求往返耗时
    pids: frozenset = frozenset()                   # 关联的本机进程 PID
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)       # 引擎特有信息(探针状态等)

    @property
    def key(self) -> str:
        return f"{self.engine}:{self.port}"

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class GpuProcess:
    """占用 GPU 显存的进程。"""

    pid: int
    name: str
    mem_bytes: Optional[int] = None


@dataclass
class GpuInfo:
    """单块 NVIDIA 显卡的实时状态。"""

    index: int
    name: str
    mem_used: int = 0
    mem_total: int = 0
    util_pct: Optional[float] = None        # 0~100
    temp_c: Optional[float] = None
    fan_pct: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    processes: list[GpuProcess] = field(default_factory=list)

    @property
    def mem_ratio(self) -> Optional[float]:
        if not self.mem_total:
            return None
        return self.mem_used / self.mem_total


@dataclass
class HardwareSnapshot:
    """一次硬件采样:GPU 列表 + 系统内存/CPU。"""

    gpus: list[GpuInfo] = field(default_factory=list)
    cpu_pct: float = 0.0
    ram_used: int = 0
    ram_total: int = 0
    swap_used: int = 0
    swap_total: int = 0
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    nvml_ok: bool = True                    # pynvml 是否可用
    error: Optional[str] = None
    timestamp: float = 0.0

    @property
    def ram_ratio(self) -> Optional[float]:
        if not self.ram_total:
            return None
        return self.ram_used / self.ram_total


class Severity(str, Enum):
    """诊断严重级别。"""

    INFO = "info"
    WARN = "warn"
    CRIT = "crit"


@dataclass
class Diagnostic:
    """一条诊断建议。"""

    severity: Severity
    title: str
    detail: str = ""
    source: str = ""          # 触发来源:引擎名 / "hardware" / "discovery"

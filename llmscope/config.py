"""集中管理所有可调参数:引擎端口、刷新间隔、诊断阈值。

零配置的默认值来自社区最常见的部署习惯;
需要微调时只改这一个文件(或 SETTINGS 单例)即可,无需动其它模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settings:
    # ---------- 引擎发现 ----------
    # 每个引擎的默认端口(可追加备选端口,如 vLLM 常见的 8080)
    default_ports: dict[str, list[int]] = field(
        default_factory=lambda: {
            "ollama": [11434],
            "vllm": [8000],
            "sglang": [30000],
        }
    )
    scan_connect_timeout: float = 0.4   # 指纹探测基础超时(秒)
    http_timeout: float = 2.5           # 常规 HTTP 请求超时(秒)
    max_process_ports: int = 16         # 进程识别得到的额外端口数上限,防止误扫

    # ---------- 刷新间隔(秒) ----------
    hardware_interval: float = 1.0      # GPU / 内存
    engine_interval: float = 2.0        # 引擎指标
    discovery_interval: float = 10.0    # 引擎发现(端口 + 进程扫描)
    diagnostic_interval: float = 4.0    # 诊断规则评估

    # ---------- Ollama 测速探针 ----------
    # 通过向后台发送一个极小的流式生成请求来估算实时 token/s。
    # 只对“已经加载”的模型探针,不会主动把冷模型拉进显存。
    ollama_probe_enabled: bool = True
    ollama_probe_interval: float = 30.0     # 两次探针的最小间隔
    ollama_probe_num_predict: int = 8       # 探针生成 token 数(越小越轻量)
    ollama_probe_timeout: float = 20.0      # 探针读超时

    # ---------- 诊断阈值 ----------
    vram_warn: float = 0.90             # 显存占用告警
    vram_crit: float = 0.97             # 显存占用危急(接近 OOM)
    gpu_temp_warn: int = 80             # GPU 温度告警(°C)
    gpu_temp_crit: int = 90
    kv_cache_warn: float = 0.90         # KV 缓存 / 令牌池占用告警
    queue_warn: int = 1                 # 排队请求数告警
    tokens_per_sec_low: dict[str, float] = field(
        default_factory=lambda: {"ollama": 5.0, "vllm": 10.0, "sglang": 10.0}
    )
    cpu_offload_warn: float = 0.15      # Ollama 权重卸载到内存的比例告警
    gpu_idle_running: float = 0.10      # 有请求但 GPU 利用率低于此值 → 瓶颈提示
    ram_warn: float = 0.90


# 全局单例;TUI 内按键可动态改写其中字段(如 ollama_probe_enabled)
SETTINGS = Settings()

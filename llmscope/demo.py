"""演示模式:无 GPU / 无引擎也能完整体验 UI 与诊断链路。

    python run.py --demo

数据由确定性正弦曲线驱动,会周期性越过各类阈值,
方便观察硬件概览、引擎表格和诊断栏的联动变化。
"""
from __future__ import annotations

import math
import time

from . import models
from .manager import EngineManager
from .models import EngineSnapshot, GpuInfo, HardwareSnapshot, ModelSlot


class DemoHardwareCollector:
    """伪造一块 24GB 显卡 + 64GB 内存的波动数据。"""

    def collect(self) -> HardwareSnapshot:
        t = time.time()
        used = 21.4e9 + 1.3e9 * math.sin(t / 40)  # 占用率 0.84~0.95,周期越过告警线
        return HardwareSnapshot(
            gpus=[
                GpuInfo(
                    index=0,
                    name="NVIDIA GeForce RTX 4090 (demo)",
                    mem_used=int(used),
                    mem_total=int(24e9),
                    util_pct=max(2.0, 65 + 30 * math.sin(t / 13)),
                    temp_c=68 + 9 * math.sin(t / 50),
                    fan_pct=55 + 15 * math.sin(t / 50),
                    power_w=280 + 90 * math.sin(t / 13),
                    power_limit_w=450,
                    processes=[
                        models.GpuProcess(pid=1234, name="python (vllm serve, demo)", mem_bytes=int(12.5e9)),
                        models.GpuProcess(pid=4321, name="ollama runner (demo)", mem_bytes=int(11.0e9)),
                    ],
                )
            ],
            cpu_pct=max(3.0, 45 + 25 * math.sin(t / 7)),
            ram_used=int(26.5e9 + 1.5e9 * math.sin(t / 30)),
            ram_total=int(64e9),
            swap_used=int(2.1e9),
            swap_total=int(8e9),
            driver_version="566.36 (demo)",
            cuda_version="12.7 (demo)",
            nvml_ok=True,
            timestamp=time.time(),
        )


class DemoEngineManager(EngineManager):
    """不真正发现 / 采集,直接产出带波动的三引擎快照。"""

    def rediscover(self) -> None:
        return None

    def collect_all(self) -> list[EngineSnapshot]:
        t = time.time()
        kv = 0.72 + 0.22 * math.sin(t / 25)  # 0.50~0.94,周期越过 0.90 阈值
        queue = max(0, int(round(1.6 + 1.8 * math.sin(t / 17 + 1.2))))
        return [
            EngineSnapshot(
                engine=models.OLLAMA,
                port=11434,
                status="online",
                version="0.5.7 (demo)",
                model="qwen2.5:14b-instruct-q4_K_M",
                models=[
                    ModelSlot(
                        name="qwen2.5:14b-instruct-q4_K_M",
                        vram_bytes=int(10.6e9),
                        size_bytes=int(11.6e9),
                        quantization="Q4_K_M",
                    )
                ],
                gen_tokens_per_sec=26.5 + 3.5 * math.sin(t / 9),
                prompt_tokens_per_sec=320 + 60 * math.sin(t / 6),
                vram_bytes=int(10.6e9),
                cpu_offload_ratio=0.086,
                extra={
                    "probe": "done",
                    "installed": ["qwen2.5:14b-instruct-q4_K_M", "llama3.1:8b"],
                },
            ),
            EngineSnapshot(
                engine=models.VLLM,
                port=8000,
                status="online",
                version="0.6.6 (demo)",
                model="meta-llama/Llama-3.1-8B-Instruct",
                gen_tokens_per_sec=345 + 45 * math.sin(t / 11),
                prompt_tokens_per_sec=1800 + 400 * math.sin(t / 8),
                kv_cache_usage=kv,
                queue_depth=queue,
                running_requests=max(0, int(4 + 3 * math.sin(t / 15))),
                vram_bytes=int(12.5e9),
                latency_ms=4.2,
            ),
            EngineSnapshot(
                engine=models.SGLANG,
                port=30000,
                status="online",
                version="0.3.6 (demo)",
                model="Qwen/Qwen2.5-32B-Instruct",
                gen_tokens_per_sec=168 + 25 * math.sin(t / 10),
                prompt_tokens_per_sec=900,
                kv_cache_usage=0.35 + 0.1 * math.sin(t / 21),
                queue_depth=0,
                running_requests=2,
                vram_bytes=int(18.8e9),
                latency_ms=3.1,
            ),
        ]

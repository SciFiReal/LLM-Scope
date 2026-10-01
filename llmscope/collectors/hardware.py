"""硬件监控采集器:NVIDIA GPU(pynvml)+ CPU / 内存(psutil)。

所有 NVML 调用都有兜底:没有 NVIDIA 驱动 / 未装 pynvml 时
快照里 gpus 为空并注明原因,其余功能照常工作。
"""
from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

import psutil

from ..models import GpuInfo, GpuProcess, HardwareSnapshot

# nvidia-ml-py(官方维护)与旧 pynvml 包都提供 pynvml 模块;
# 旧包会发 FutureWarning,这里统一压制,两者皆可用
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore[assignment]


def _safe(fn: Callable, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _pid_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name()
    except Exception:
        return str(pid)


class HardwareCollector:
    """每次 collect() 返回一份 HardwareSnapshot。"""

    def __init__(self) -> None:
        self._handles: list = []
        self._driver: Optional[str] = None
        self._cuda: Optional[str] = None
        self._init_error: Optional[str] = None

        if pynvml is None:
            self._init_error = "pynvml 未安装(pip install pynvml)"
        else:
            try:
                pynvml.nvmlInit()
                self._handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
                self._driver = _safe(lambda: str(pynvml.nvmlSystemGetDriverVersion()))
                self._cuda = _safe(lambda: str(pynvml.nvmlSystemGetCudaVersion()))
            except Exception as exc:
                self._init_error = f"NVML 初始化失败:{exc}"

        # 预热 psutil 的 CPU 采样,避免首次采集读到 0
        psutil.cpu_percent(interval=None)

    @property
    def available(self) -> bool:
        return bool(self._handles)

    def collect(self) -> HardwareSnapshot:
        gpus = [self._collect_gpu(i, h) for i, h in enumerate(self._handles)]
        vm, sm = psutil.virtual_memory(), psutil.swap_memory()
        return HardwareSnapshot(
            gpus=gpus,
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_used=vm.used,
            ram_total=vm.total,
            swap_used=sm.used,
            swap_total=sm.total,
            driver_version=self._driver,
            cuda_version=self._cuda,
            nvml_ok=self.available,
            error=self._init_error if not self.available else None,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    def _collect_gpu(self, index: int, handle) -> GpuInfo:
        info = GpuInfo(
            index=index,
            name=_safe(lambda: str(pynvml.nvmlDeviceGetName(handle))) or f"GPU {index}",
        )
        mem = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        if mem:
            info.mem_used, info.mem_total = int(mem.used), int(mem.total)
        util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        info.util_pct = float(util) if util is not None else None
        info.temp_c = _safe(
            lambda: float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        )
        info.fan_pct = _safe(lambda: float(pynvml.nvmlDeviceGetFanSpeed(handle)))
        power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        info.power_w = power / 1000 if power else None
        limit = _safe(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        info.power_limit_w = limit / 1000 if limit else None
        info.processes = self._gpu_processes(handle)
        return info

    @staticmethod
    def _gpu_processes(handle) -> list[GpuProcess]:
        """合并计算 / 图形两类占用;Windows 的 WDDM 驱动多在 graphics 列表。"""
        found: dict[int, GpuProcess] = {}
        for fn in (
            pynvml.nvmlDeviceGetComputeRunningProcesses,
            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            try:
                for p in fn(handle):
                    mem = int(p.usedGpuMemory) if p.usedGpuMemory else None
                    if p.pid in found:
                        if mem and (found[p.pid].mem_bytes or 0) < mem:
                            found[p.pid].mem_bytes = mem
                    else:
                        found[p.pid] = GpuProcess(pid=p.pid, name=_pid_name(p.pid), mem_bytes=mem)
            except Exception:
                continue
        return list(found.values())

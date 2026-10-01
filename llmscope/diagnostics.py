"""诊断规则引擎:硬件快照 + 引擎快照 → 建议列表。

新增规则:写一个函数 (hardware, engines, settings) -> list[Diagnostic],
加上 @rule 装饰器即可,无需改动任何调用方。
所有阈值集中在 config.Settings,便于按个人硬件调优。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .config import SETTINGS, Settings
from .models import Diagnostic, EngineSnapshot, HardwareSnapshot, Severity
from .utils import fmt_pct

RULES: list[Callable] = []


def rule(fn: Callable) -> Callable:
    """规则注册装饰器。"""
    RULES.append(fn)
    return fn


def evaluate(
    hardware: Optional[HardwareSnapshot],
    engines: List[EngineSnapshot],
    settings: Settings = SETTINGS,
) -> list[Diagnostic]:
    """执行全部规则,按严重级别排序(CRIT > WARN > INFO)。"""
    diags: list[Diagnostic] = []
    for fn in RULES:
        try:
            diags.extend(fn(hardware, engines, settings))
        except Exception:
            continue  # 单条规则异常不影响整体
    order = {Severity.CRIT: 0, Severity.WARN: 1, Severity.INFO: 2}
    diags.sort(key=lambda d: order[d.severity])
    return diags


# ======================================================================
# 发现层
# ======================================================================
@rule
def rule_no_engine(hardware, engines, s):
    if engines:
        return []
    return [
        Diagnostic(
            Severity.INFO,
            "未发现运行中的推理引擎",
            "启动 Ollama(ollama serve)、vLLM 或 SGLang 后会自动接入;"
            "本机没有引擎时可用 python run.py --demo 查看演示数据。",
            source="discovery",
        )
    ]


@rule
def rule_engine_down(hardware, engines, s):
    out = []
    for e in engines:
        if e.status == "unreachable":
            out.append(
                Diagnostic(
                    Severity.WARN,
                    f"{e.engine}({e.addr}) 失联",
                    f"最近错误:{e.error or '未知'}。确认服务进程存活、端口未变更;"
                    "vLLM/SGLang 冷启动加载权重期间响应慢属正常现象。",
                    source=e.engine,
                )
            )
    return out


@rule
def rule_ollama_idle(hardware, engines, s):
    """Ollama 在线但显存里没有模型 → 告诉用户怎么让界面"活"起来。"""
    for e in engines:
        if e.engine == "ollama" and e.status == "online" and e.model is None:
            return [
                Diagnostic(
                    Severity.INFO,
                    "Ollama 在线,尚未加载模型",
                    "在任意终端运行 ollama run <模型名> 并发一句话,"
                    "这里就会自动出现显存占用、生成速度与 offload 提示;"
                    "也可用 python run.py --demo 先看演示效果。",
                    source=e.engine,
                )
            ]
    return []


# ======================================================================
# 硬件层
# ======================================================================
@rule
def rule_vram_pressure(hardware, engines, s):
    if not hardware:
        return []
    out = []
    for g in hardware.gpus:
        r = g.mem_ratio
        if r is None:
            continue
        if r >= s.vram_crit:
            out.append(
                Diagnostic(
                    Severity.CRIT,
                    f"GPU{g.index} 显存接近打满({fmt_pct(r)})",
                    "随时可能 OOM 或触发 KV 缓存抢占:vLLM 调低 --gpu-memory-utilization "
                    "或换 AWQ/GPTQ 量化模型;SGLang 调低 --mem-fraction-static;"
                    "Ollama 设置 OLLAMA_MAX_LOADED_MODELS=1、缩短 OLLAMA_KEEP_ALIVE,"
                    "或换 Q4 量化小模型。",
                    source="hardware",
                )
            )
        elif r >= s.vram_warn:
            out.append(
                Diagnostic(
                    Severity.WARN,
                    f"GPU{g.index} 显存占用偏高({fmt_pct(r)})",
                    "新负载可能失败:vLLM/SGLang 预留空间调低;"
                    "Ollama 减少同时加载的模型数或换量化模型。",
                    source="hardware",
                )
            )
    return out


@rule
def rule_gpu_temp(hardware, engines, s):
    if not hardware:
        return []
    out = []
    for g in hardware.gpus:
        if g.temp_c is None:
            continue
        if g.temp_c >= s.gpu_temp_crit:
            out.append(
                Diagnostic(
                    Severity.CRIT,
                    f"GPU{g.index} 温度 {g.temp_c:.0f}°C,接近保护阈值",
                    "高温会强制降频、直接拉低 token/s:清理风扇与散热器积灰、"
                    "改善机箱风道,必要时用 nvidia-smi -pl 限制功耗。",
                    source="hardware",
                )
            )
        elif g.temp_c >= s.gpu_temp_warn:
            out.append(
                Diagnostic(
                    Severity.WARN,
                    f"GPU{g.index} 温度偏高({g.temp_c:.0f}°C)",
                    "检查风扇转速与机箱风道;长时间高负载建议清理积灰。",
                    source="hardware",
                )
            )
    return out


@rule
def rule_ram_pressure(hardware, engines, s):
    if not hardware:
        return []
    r = hardware.ram_ratio
    if r is None or r < s.ram_warn:
        return []
    return [
        Diagnostic(
            Severity.WARN,
            f"系统内存占用 {fmt_pct(r)}",
            "模型加载与上下文缓存都吃内存:vLLM/SGLang 可减小 --swap-space;"
            "关闭其它占内存的进程,避免 swap 拖垮首 token 延迟。",
            source="hardware",
        )
    ]


@rule
def rule_gpu_idle(hardware, engines, s):
    """有请求在跑但 GPU 几乎空转 → 瓶颈不在 GPU。"""
    if not hardware or not hardware.gpus:
        return []
    util = hardware.gpus[0].util_pct
    if util is None:
        return []
    busy = [e for e in engines if e.status == "online" and (e.running_requests or 0) > 0]
    if not busy or util >= s.gpu_idle_running * 100:
        return []
    names = "、".join(e.engine for e in busy)
    return [
        Diagnostic(
            Severity.INFO,
            f"有请求在跑但 GPU 利用率仅 {util:.0f}%({names})",
            "瓶颈通常不在 GPU:小 batch 长尾、CPU 预处理/分词,"
            "或 Ollama 模型部分层在 CPU(结合 offload 提示确认)。",
            source="hardware",
        )
    ]


# ======================================================================
# 引擎层
# ======================================================================
@rule
def rule_kv_cache_pressure(hardware, engines, s):
    out = []
    for e in engines:
        kv = e.kv_cache_usage
        if e.status != "online" or kv is None or kv < s.kv_cache_warn:
            continue
        if e.engine == "vllm":
            tip = (
                "vLLM KV 缓存吃紧:增大 --gpu-memory-utilization、"
                "减小 --max-num-seqs 或 --max-model-len;"
                "持续打满会抢占(preempt)请求、拖慢吞吐。"
            )
        else:
            tip = (
                "SGLang 令牌池吃紧:调高 --mem-fraction-static、"
                "缩短 --context-length 或降低并发;持续打满会排队等待。"
            )
        out.append(
            Diagnostic(Severity.WARN, f"{e.engine} KV缓存占用 {fmt_pct(kv)}", tip, source=e.engine)
        )
    return out


@rule
def rule_queue_backlog(hardware, engines, s):
    out = []
    for e in engines:
        q = e.queue_depth
        if e.status != "online" or q is None or q < s.queue_warn:
            continue
        if e.engine == "vllm":
            tip = (
                "请求排队积压:提高 --max-num-seqs、增大 --gpu-memory-utilization,"
                "或加卡(--tensor-parallel-size N)/ 换更小模型提升吞吐。"
            )
        else:
            tip = "请求在队列等待:提高并发上限,加卡(--tp-size N)或换更小模型。"
        out.append(
            Diagnostic(Severity.WARN, f"{e.engine} 有 {q} 个请求排队", tip, source=e.engine)
        )
    return out


@rule
def rule_preempted(hardware, engines, s):
    out = []
    for e in engines:
        n = e.preempted_requests
        if e.status != "online" or not n:
            continue
        out.append(
            Diagnostic(
                Severity.WARN,
                f"{e.engine} 检测到被抢占请求({n})",
                "KV 缓存不足导致请求换出重算:vLLM 降低并发(--max-num-seqs)、"
                "缩短 --max-model-len 或提高 --gpu-memory-utilization。",
                source=e.engine,
            )
        )
    return out


@rule
def rule_low_throughput(hardware, engines, s):
    out = []
    for e in engines:
        if e.status != "online" or e.gen_tokens_per_sec is None:
            continue
        threshold = s.tokens_per_sec_low.get(e.engine)
        if threshold is None or e.gen_tokens_per_sec >= threshold:
            continue
        # 空闲(无在跑请求)时吞吐为 0 属正常,跳过
        if e.engine != "ollama" and not (e.running_requests or 0):
            continue
        if e.engine == "ollama":
            tip = (
                "生成速度偏低:多半是显存不足、部分层退回 CPU(结合 offload 提示确认)。"
                "换 Q4 量化小模型、设置 OLLAMA_NUM_PARALLEL=1、"
                "开启 OLLAMA_FLASH_ATTENTION=1。"
            )
        else:
            tip = "吞吐偏低且有请求在跑:检查 GPU 是否降频(看温度)、CPU 预处理是否瓶颈、batch 是否过小。"
        out.append(
            Diagnostic(
                Severity.WARN,
                f"{e.engine} 生成速度仅 {e.gen_tokens_per_sec:.1f} tok/s",
                tip,
                source=e.engine,
            )
        )
    return out


@rule
def rule_cpu_offload(hardware, engines, s):
    out = []
    for e in engines:
        r = e.cpu_offload_ratio
        if e.status != "online" or r is None or r < s.cpu_offload_warn:
            continue
        out.append(
            Diagnostic(
                Severity.WARN,
                f"{e.engine} 约 {fmt_pct(r)} 权重在内存(CPU offload)",
                "显存放不下完整模型,推理会显著变慢:换更小参数量 / 更低量化"
                "(如 Q4_K_M)的模型,或 OLLAMA_NUM_PARALLEL=1、"
                "OLLAMA_MAX_LOADED_MODELS=1 释放显存后再载入大模型。",
                source=e.engine,
            )
        )
    return out

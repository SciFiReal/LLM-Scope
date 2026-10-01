"""UI 组件:纯渲染函数 + Textual Widget。

渲染函数(render_hardware / engine_cells / render_diagnostics)同时被
TUI 与 --once 文本报告复用,保证两种出口的展示一致。
"""
from __future__ import annotations

from typing import Optional

from rich.console import Group
from rich.table import Table as RichTable
from rich.text import Text
from textual.widgets import DataTable, Static

from ..config import SETTINGS
from ..models import Diagnostic, EngineSnapshot, HardwareSnapshot, Severity
from ..utils import fmt_num, fmt_pct, human_bytes, short_gpu_name, trunc

SEV_STYLE = {Severity.CRIT: "bold red", Severity.WARN: "yellow", Severity.INFO: "cyan"}
SEV_ICON = {Severity.CRIT: "✖", Severity.WARN: "▲", Severity.INFO: "ℹ"}


def _dash() -> Text:
    """结构性无数据的占位(如 Ollama 不提供 KV%/队列),用暗色弱化。"""
    return Text("—", style="dim")


def _bar(ratio: Optional[float], width: int = 14) -> Text:
    """htop 风格的字符比例条;阈值颜色:绿 / 黄 / 红。"""
    if ratio is None:
        return Text("—", style="dim")
    ratio = max(0.0, min(1.0, ratio))
    color = "red" if ratio >= 0.9 else "yellow" if ratio >= 0.75 else "green"
    filled = round(ratio * width)
    return Text("█" * filled + "░" * (width - filled), style=color)


# ======================================================================
# 顶部:硬件概览
# ======================================================================
def render_hardware(hw: Optional[HardwareSnapshot], show_processes: bool = False) -> Group:
    table = RichTable(
        box=None, padding=(0, 1), show_header=True,
        header_style="bold dim", expand=True,
    )
    for col in ("GPU", "显存", "占用", "利用率", "温度", "风扇", "功率", "GPU进程"):
        table.add_column(col)

    if hw is None:
        return Group(Text("正在采集硬件信息…", style="dim"))
    if not hw.gpus:
        note = hw.error or "未检测到 NVIDIA GPU(需要 NVIDIA 驱动 + pynvml)"
        table.add_row(Text("GPU", style="bold"), Text(note, style="yellow"))
    else:
        for g in hw.gpus:
            temp = Text("—", style="dim")
            if g.temp_c is not None:
                if g.temp_c >= SETTINGS.gpu_temp_crit:
                    style = "bold red"
                elif g.temp_c >= SETTINGS.gpu_temp_warn:
                    style = "yellow"
                else:
                    style = "green"
                temp = Text(f"{g.temp_c:.0f}°C", style=style)
            util = fmt_pct(g.util_pct / 100 if g.util_pct is not None else None)
            power = "—" if g.power_w is None else f"{g.power_w:.0f}W"
            if g.power_w is not None and g.power_limit_w:
                power += f"/{g.power_limit_w:.0f}W"
            table.add_row(
                Text(f"GPU{g.index} {short_gpu_name(g.name)}", style="bold"),
                f"{human_bytes(g.mem_used)} / {human_bytes(g.mem_total)}",
                _bar(g.mem_ratio),
                util,
                temp,
                fmt_pct(g.fan_pct / 100 if g.fan_pct is not None else None),
                power,
                f"{len(g.processes)} 个",
            )

    ram_ratio = hw.ram_ratio
    table.add_row(
        Text("系统", style="bold"),
        f"内存 {human_bytes(hw.ram_used)} / {human_bytes(hw.ram_total)}",
        _bar(ram_ratio),
        f"CPU {hw.cpu_pct:.0f}%",
        f"交换 {human_bytes(hw.swap_used)}",
        "",
        "",
        "",
    )
    if show_processes:
        return Group(table, _process_table(hw))
    return Group(table)


def _process_table(hw: HardwareSnapshot) -> RichTable:
    """GPU 进程明细:按显存占用降序列出各进程(驱动未报告的排最后)。"""
    t = RichTable(
        box=None, padding=(0, 1), show_header=True,
        header_style="bold dim", expand=True,
    )
    for col in ("GPU", "PID", "进程", "显存"):
        t.add_column(col)

    procs: list[tuple[int, object]] = []
    for g in hw.gpus:
        for p in g.processes:
            procs.append((g.index, p))
    procs.sort(key=lambda ip: (ip[1].mem_bytes is None, -(ip[1].mem_bytes or 0)))

    if not procs:
        t.add_row(
            Text("—", style="dim"),
            Text("驱动未报告任何 GPU 进程", style="dim"),
            "", "",
        )
        return t

    shown = procs[:15]
    for gpu_index, p in shown:
        mem = (
            Text("未报告", style="dim")
            if p.mem_bytes is None
            else human_bytes(p.mem_bytes)
        )
        t.add_row(f"GPU{gpu_index}", str(p.pid), trunc(p.name, 44), mem)
    if len(procs) > len(shown):
        t.add_row(
            "",
            Text(
                f"… 其余 {len(procs) - len(shown)} 个多为系统图形进程"
                "(Windows WDDM 驱动常不报告单进程显存)",
                style="dim",
            ),
            "", "",
        )
    return t


class HardwareOverview(Static):
    """顶部硬件概览面板。"""

    def show_snapshot(
        self, hw: Optional[HardwareSnapshot], show_processes: bool = False
    ) -> None:
        if hw and hw.driver_version:
            sub = f"驱动 {hw.driver_version}"
            if hw.cuda_version:
                sub += f" · CUDA {hw.cuda_version}"
            # 简称截掉的品牌/系列信息,完整显卡名放在副标题里备查
            fulls = [
                f"GPU{g.index}: {g.name}"
                for g in hw.gpus
                if short_gpu_name(g.name) != g.name
            ]
            if fulls:
                sub += " · " + "; ".join(fulls)
            self.border_subtitle = sub
        self.update(render_hardware(hw, show_processes))


# ======================================================================
# 中部:引擎列表
# ======================================================================
ENGINE_COLUMNS = [
    "状态", "引擎", "地址", "版本", "模型",
    "生成", "预填", "KV%", "队列", "运行", "显存", "备注",
]


def engine_cells(e: EngineSnapshot) -> list:
    """把 EngineSnapshot 转成一行单元格(富文本),供 TUI 与报告共用。"""
    if e.status == "online":
        status = Text("● 在线", style="green")
    else:
        status = Text("● 失联", style="red")

    # 无数据的单元格一律用暗色"—":区分"引擎不支持"(如 Ollama 无 KV%/队列)
    # 和有数据;支持的引擎拿到数据后自动点亮
    gen = _dash() if e.gen_tokens_per_sec is None else Text(f"{e.gen_tokens_per_sec:.1f}")
    prefill = (
        _dash() if e.prompt_tokens_per_sec is None else Text(f"{e.prompt_tokens_per_sec:.1f}")
    )

    kv = _dash()
    if e.kv_cache_usage is not None:
        style = ""
        if e.kv_cache_usage >= SETTINGS.kv_cache_warn:
            style = "bold red"
        elif e.kv_cache_usage >= 0.75:
            style = "yellow"
        kv = Text(fmt_pct(e.kv_cache_usage), style=style or "green")

    if e.queue_depth is None:
        queue: object = _dash()
    elif e.queue_depth >= SETTINGS.queue_warn:
        queue = Text(str(e.queue_depth), style="yellow")
    else:
        queue = Text(str(e.queue_depth), style="green")

    run = _dash() if e.running_requests is None else Text(str(int(e.running_requests)))

    # 模型列:Ollama 在线但没加载任何模型时,给出明确状态而不是空 dash
    model_disp = trunc(e.model, 26)
    if e.status == "online" and e.model is None and e.engine == "ollama":
        model_disp = "未加载"

    notes: list[str] = []
    if e.error:
        notes.append(trunc(e.error, 44))
    if e.engine == "ollama":
        if e.status == "online" and e.model is None:
            notes.append("空闲 · 任意终端跑一次 ollama run 即显示实时速度")
        probe = e.extra.get("probe")
        if probe == "disabled":
            notes.append("测速已关闭(按 o 开启)")
        elif probe == "cached":
            notes.append("测速间隔中")
        elif isinstance(probe, str) and probe.startswith("failed"):
            notes.append("测速失败:" + trunc(probe[7:], 30))
        r = e.cpu_offload_ratio
        if r is not None and r > 0.02:
            notes.append(f"offload {fmt_pct(r)}")
        installed = e.extra.get("installed")
        if installed is not None:
            notes.append(f"已装 {len(installed)} 个模型")
    if e.preempted_requests:
        notes.append(f"抢占 {e.preempted_requests}")

    return [
        status,
        e.engine,
        e.addr,
        e.version or _dash(),
        model_disp,
        gen,
        prefill,
        kv,
        queue,
        run,
        human_bytes(e.vram_bytes),
        " · ".join(notes) or "—",
    ]


class EngineTable(DataTable):
    """中部引擎列表(htable 风格表格),按 引擎:端口 稳定更新行。"""

    def on_mount(self) -> None:
        self.zebra_stripes = True
        self.cursor_type = "row"
        self._row_keys: set[str] = set()
        self._order: list[str] = []          # 行序号 → 引擎 key,供详情页定位
        for label in ENGINE_COLUMNS:
            self.add_column(label, key=label)

    @property
    def selected_key(self) -> Optional[str]:
        """当前光标所在行的引擎 key;表格为空返回 None。"""
        if not self._order:
            return None
        idx = self.cursor_row
        if 0 <= idx < len(self._order):
            return self._order[idx]
        return None

    def show_snapshots(self, snaps: list[EngineSnapshot]) -> None:
        seen: set[str] = set()
        for e in snaps:
            rk = e.key
            cells = engine_cells(e)
            if rk in self._row_keys:
                for col, val in zip(ENGINE_COLUMNS, cells):
                    try:
                        self.update_cell(rk, col, val, update_width=True)
                    except Exception:
                        pass
            else:
                self.add_row(*cells, key=rk)
                self._row_keys.add(rk)
            seen.add(rk)
        for gone in self._row_keys - seen:
            try:
                self.remove_row(gone)
            except Exception:
                pass
            self._row_keys.discard(gone)
        self._order = [e.key for e in snaps]
        self.border_subtitle = f"{len(snaps)} 个引擎实例 · 生成/预填单位 tok/s · r 刷新"


# ======================================================================
# 底部:诊断提示栏
# ======================================================================
def render_diagnostics(diags: list[Diagnostic], max_items: int = 8) -> Group:
    if not diags:
        return Group(Text("✓ 暂无异常,系统运行正常", style="green dim"))
    parts: list[Text] = []
    for d in diags[:max_items]:
        parts.append(
            Text(f"{SEV_ICON[d.severity]} [{d.severity.value.upper()}] {d.title}",
                 style=SEV_STYLE[d.severity])
        )
        line = Text()
        if d.detail:
            line.append(f"↳ {d.detail}", style="dim")
        if d.source:
            line.append(f"  · 来源 {d.source}", style="italic dim")
        if line.plain:
            parts.append(line)
    return Group(*parts)


class DiagnosticPanel(Static):
    """底部诊断提示面板。"""

    def show_diagnostics(self, diags: list[Diagnostic]) -> None:
        self.update(render_diagnostics(diags))


# ======================================================================
# 详情页:单个引擎的全部字段 + /metrics 原文(可选)
# 注:整页用单张两列表格渲染(Textual 的 Static 对"文本+表格混排"
# 的 Group 存在行高测量偏差),小节标题以加粗行呈现。
# ======================================================================
def render_engine_detail(snap: EngineSnapshot, show_raw: bool = False) -> Group:
    online = snap.status == "online"
    head = Text(snap.engine, style="bold")
    head.append(f"  {snap.addr}", style="cyan")
    head.append("   ● 在线", style="green") if online else head.append(
        "   ● 失联", style="red"
    )

    t = RichTable(box=None, padding=(0, 1), show_header=False, expand=True)
    t.add_column(min_width=14)

    def row(label: str, value: object) -> None:
        t.add_row(Text(label, style="bold dim"), value)

    def section(title: str) -> None:
        t.add_row("", "")
        t.add_row(Text(title, style="bold yellow"), "")

    def num_or_dash(v: Optional[float], suffix: str = "") -> object:
        return _dash() if v is None else Text(f"{v:.1f}{suffix}")

    def int_or_dash(v: Optional[int]) -> object:
        return _dash() if v is None else Text(str(int(v)))

    row("版本", snap.version or _dash())
    row(
        "采集延迟",
        _dash() if snap.latency_ms is None else Text(f"{snap.latency_ms:.0f} ms"),
    )
    if snap.pids:
        row("关联进程", ", ".join(str(p) for p in sorted(snap.pids)))
    if snap.error:
        row("最近错误", Text(trunc(snap.error, 90), style="red"))

    section("吞吐与请求")
    row("生成 tok/s", num_or_dash(snap.gen_tokens_per_sec))
    row("预填 tok/s", num_or_dash(snap.prompt_tokens_per_sec))
    row("运行中请求", int_or_dash(snap.running_requests))
    row("排队请求", int_or_dash(snap.queue_depth))
    row("被抢占请求", int_or_dash(snap.preempted_requests))
    row(
        "KV缓存占用",
        fmt_pct(snap.kv_cache_usage) if snap.kv_cache_usage is not None else _dash(),
    )

    section("资源")
    row("引擎显存", human_bytes(snap.vram_bytes))
    row(
        "CPU offload" if snap.engine == "ollama" else "权重卸载比例",
        fmt_pct(snap.cpu_offload_ratio) if snap.cpu_offload_ratio is not None else _dash(),
    )

    if snap.models:
        section("已加载模型")
        for i, m in enumerate(snap.models):
            row(f"模型[{i}]", trunc(m.name, 48))
            row(
                "  显存/体积",
                f"{human_bytes(m.vram_bytes)} / {human_bytes(m.size_bytes)}",
            )
            row("  量化", m.quantization or "—")

    extra_rows = [
        (k, trunc(str(v), 90)) for k, v in snap.extra.items() if k != "metrics_text"
    ]
    if extra_rows:
        section("引擎附加信息")
        for k, v in extra_rows:
            row(k, v)

    raw = snap.extra.get("metrics_text")
    if raw:
        section("原始数据")
        if show_raw:
            t.add_row(Text("/metrics 原始输出", style="bold dim"), "")
            t.add_row("", Text(str(raw)))
        else:
            row("提示", Text("按 m 查看 /metrics 原始输出", style="dim"))

    return Group(head, Text(), t)

"""TUI 应用:htop 风格三栏布局。

顶部硬件概览 / 中部引擎列表 / 底部诊断提示栏;
所有阻塞调用(requests、NVML)都通过 asyncio.to_thread 丢进线程池,
事件循环只做渲染,保证界面流畅。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from ..config import SETTINGS
from ..diagnostics import evaluate
from ..manager import EngineManager
from ..models import EngineSnapshot, HardwareSnapshot
from .screens import EngineDetailScreen
from .widgets import DiagnosticPanel, EngineTable, HardwareOverview


class LlmScopeApp(App[None]):
    TITLE = "LLM-Scope"
    SUB_TITLE = "本地 LLM 引擎监控"

    CSS = """
    #hardware {
        height: auto;
        max-height: 42%;
        border: round $primary;
        padding: 0 1;
    }
    #engines {
        height: 1fr;
        min-height: 8;
        border: round $primary;
    }
    #diagnostics {
        height: 34%;
        min-height: 7;
        border: round $warning;
        padding: 0 1;
    }
    #detail-wrap {
        width: 92%;
        height: 88%;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "退出"),
        Binding("r", "refresh", "刷新"),
        Binding("p", "pause", "暂停"),
        Binding("o", "probe", "测速"),
        Binding("g", "gpuproc", "GPU进程"),
        # priority=True:焦点在 DataTable 上时,表格内置的"选中行"动作
        # 会消费回车键,必须用优先级绑定才能先于它触发
        Binding("enter", "detail", "详情", priority=True),
    ]

    def __init__(self, demo: bool = False) -> None:
        super().__init__()
        self.demo = demo
        self._paused = False
        self._show_proc = False
        self._hw: Optional[HardwareSnapshot] = None
        self._snaps: list[EngineSnapshot] = []
        if demo:
            from ..demo import DemoEngineManager, DemoHardwareCollector

            self.manager: EngineManager = DemoEngineManager()
            self.hw_collector = DemoHardwareCollector()
        else:
            from ..collectors.hardware import HardwareCollector

            self.manager = EngineManager()
            self.hw_collector = HardwareCollector()

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield HardwareOverview(id="hardware")
        yield EngineTable(id="engines")
        yield DiagnosticPanel(id="diagnostics")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(SETTINGS.hardware_interval, self._tick_hardware)
        self.set_interval(SETTINGS.engine_interval, self._tick_engines)
        self.set_interval(SETTINGS.discovery_interval, self._tick_discovery)
        self.set_interval(SETTINGS.diagnostic_interval, self._tick_diagnostics)
        self.run_worker(self._initial_load(), name="init", exit_on_error=False)

    async def _initial_load(self) -> None:
        """启动 / 手动刷新时:先发现,再并行拉硬件与引擎指标,最后评估诊断。"""
        await self._tick_discovery()
        await asyncio.gather(self._tick_hardware(), self._tick_engines())
        await self._tick_diagnostics()

    # ---- 周期任务 ----
    async def _tick_hardware(self) -> None:
        if self._paused:
            return
        hw = await asyncio.to_thread(self.hw_collector.collect)
        self._hw = hw
        self.query_one(HardwareOverview).show_snapshot(
            hw, show_processes=self._show_proc
        )

    async def _tick_discovery(self) -> None:
        if self._paused:
            return
        await asyncio.to_thread(self.manager.rediscover)
        if self.demo:
            self.sub_title = "演示模式(--demo)"
        else:
            self.sub_title = f"已接入 {len(self.manager.entries)} 个引擎实例"

    async def _tick_engines(self) -> None:
        if self._paused:
            return
        snaps = await asyncio.to_thread(self.manager.collect_all)
        self._snaps = snaps
        self.query_one(EngineTable).show_snapshots(snaps)

    async def _tick_diagnostics(self) -> None:
        if self._paused:
            return
        diags = evaluate(self._hw, self._snaps)
        self.query_one(DiagnosticPanel).show_diagnostics(diags)

    # ---- 按键 ----
    def action_refresh(self) -> None:
        self.run_worker(self._initial_load(), name="refresh", exit_on_error=False)

    def action_pause(self) -> None:
        self._paused = not self._paused
        self.notify(
            "已暂停采集,按 p 恢复" if self._paused else "已恢复采集",
            title="暂停",
        )

    def action_probe(self) -> None:
        SETTINGS.ollama_probe_enabled = not SETTINGS.ollama_probe_enabled
        state = "开启" if SETTINGS.ollama_probe_enabled else "关闭"
        self.notify(f"Ollama 后台测速已{state}", title="测速")

    def action_detail(self) -> None:
        """回车:进入当前选中引擎的详情页。"""
        # 优先级绑定会穿透模态层触发,详情页已打开时直接忽略
        if isinstance(self.screen, EngineDetailScreen):
            return
        key = self.query_one(EngineTable).selected_key
        if not key:
            self.notify("当前没有引擎行", title="详情")
            return
        self.push_screen(EngineDetailScreen(key))

    def action_gpuproc(self) -> None:
        """g:展开 / 收起 GPU 进程显存明细。"""
        self._show_proc = not self._show_proc
        self.query_one(HardwareOverview).show_snapshot(
            self._hw, show_processes=self._show_proc
        )
        state = "展开" if self._show_proc else "收起"
        self.notify(f"GPU 进程明细已{state}", title="GPU进程")

    def on_unmount(self) -> None:
        self.manager.close()

"""命令行入口。

    python run.py                # 启动 TUI
    python run.py --demo         # 使用演示数据(无需 GPU / 引擎)
    python run.py --once         # 单次采集,输出文本报告后退出(便于脚本 / CI)
    python -m llmscope           # 等价于 python run.py
"""
from __future__ import annotations

import argparse
import sys

from rich import box
from rich.console import Console
from rich.table import Table as RichTable

from . import __version__
from .config import SETTINGS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-scope", description="轻量零配置的本地 LLM 多引擎监控工具"
    )
    p.add_argument("--demo", action="store_true", help="使用内置演示数据(无需 GPU / 引擎)")
    p.add_argument("--once", action="store_true", help="单次采集并输出报告后退出(不进 TUI)")
    p.add_argument("--interval", type=float, default=None, help="覆盖引擎指标刷新间隔(秒)")
    p.add_argument("--version", action="version", version=f"LLM-Scope {__version__}")
    return p


def _build_stack(demo: bool):
    """返回 (引擎管理器, 硬件采集器)。"""
    if demo:
        from .demo import DemoEngineManager, DemoHardwareCollector

        return DemoEngineManager(), DemoHardwareCollector()
    from .collectors.hardware import HardwareCollector
    from .manager import EngineManager

    return EngineManager(), HardwareCollector()


def run_once(demo: bool) -> int:
    """单次采集模式:硬件概览 + 引擎表格 + 诊断建议,打印后退出。"""
    from .diagnostics import evaluate
    from .ui.widgets import ENGINE_COLUMNS, engine_cells, render_diagnostics, render_hardware

    manager, hw_collector = _build_stack(demo)
    try:
        manager.rediscover()
        snaps = manager.collect_all()
        hw = hw_collector.collect()
    finally:
        manager.close()

    console = Console()
    mode = "演示数据" if demo else "实时采集"
    console.print(f"[bold]LLM-Scope[/] v{__version__} — 单次采集报告({mode})\n")
    console.print(render_hardware(hw))

    console.print()
    if snaps:
        table = RichTable(*ENGINE_COLUMNS, box=box.SIMPLE_HEAVY, expand=True)
        for s in snaps:
            table.add_row(*engine_cells(s))
        console.print(table)
    else:
        console.print("[dim]未发现运行中的推理引擎。[/]")

    console.print()
    console.print(render_diagnostics(evaluate(hw, snaps)))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval is not None:
        SETTINGS.engine_interval = max(0.5, args.interval)
    if args.once:
        return run_once(args.demo)

    from .ui.app import LlmScopeApp

    try:
        LlmScopeApp(demo=args.demo).run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

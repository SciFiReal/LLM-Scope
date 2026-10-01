"""引擎详情页:引擎表格里按回车进入,esc 返回,m 切换 /metrics 原文。

详情页每秒自动刷新,数据来自 App 缓存的最新一轮 EngineSnapshot;
引擎下线时给出提示,不会崩掉。
"""
from __future__ import annotations

from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from .widgets import render_engine_detail


class EngineDetailScreen(ModalScreen[None]):
    """单个引擎的全部原始指标(模态浮层)。"""

    BINDINGS = [
        Binding("escape", "dismiss", "返回"),
        Binding("q", "dismiss", "返回", show=False),
        Binding("m", "raw", "原始/metrics", show=False),
    ]

    def __init__(self, engine_key: str) -> None:
        super().__init__()
        self._key = engine_key      # 形如 "ollama:11434"
        self._show_raw = False

    def compose(self):
        yield VerticalScroll(
            Static("正在加载…", id="detail-body"),
            id="detail-wrap",
        )

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        snaps = getattr(self.app, "_snaps", [])
        snap = next((s for s in snaps if s.key == self._key), None)
        body = self.query_one("#detail-body", Static)
        if snap is None:
            body.update("该引擎已下线,按 esc 返回")
            return
        body.update(render_engine_detail(snap, show_raw=self._show_raw))

    def action_raw(self) -> None:
        self._show_raw = not self._show_raw
        self._refresh()

    def action_dismiss(self) -> None:
        self.dismiss()

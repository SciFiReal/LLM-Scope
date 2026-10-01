"""导出演示模式的 TUI 截图(SVG),供 README / 文档预览。

    .venv/Scripts/python scripts/screenshot.py
输出(docs/ 目录,随仓库提交):
    docs/screenshot.svg          主界面(GPU 进程明细展开)
    docs/screenshot_detail.svg   引擎详情页(回车进入的浮层)
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llmscope.ui.app import LlmScopeApp

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"


async def main() -> None:
    DOCS.mkdir(exist_ok=True)
    app = LlmScopeApp(demo=True)
    async with app.run_test(size=(130, 34)) as pilot:
        await pilot.pause()
        for _ in range(6):
            await pilot.pause(0.25)
        app.action_gpuproc()          # 展开 GPU 进程明细
        await pilot.pause(0.3)
        (DOCS / "screenshot.svg").write_text(
            app.export_screenshot(), encoding="utf-8"
        )
        app.action_detail()           # 打开第一个引擎的详情页
        await pilot.pause(0.5)
        (DOCS / "screenshot_detail.svg").write_text(
            app.export_screenshot(), encoding="utf-8"
        )
        print("已导出 docs/screenshot.svg 与 docs/screenshot_detail.svg")


if __name__ == "__main__":
    asyncio.run(main())

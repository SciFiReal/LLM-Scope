"""冒烟测试:演示数据管线 + 无头 TUI 渲染。

    .venv/Scripts/python -m pytest tests -q
或直接:.venv/Scripts/python tests/test_smoke.py
"""
import asyncio
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDemoPipeline(unittest.TestCase):
    def test_demo_snapshots(self):
        from llmscope.demo import DemoEngineManager, DemoHardwareCollector
        from llmscope.diagnostics import evaluate

        snaps = DemoEngineManager().collect_all()
        self.assertEqual(len(snaps), 3)
        self.assertTrue(all(s.status == "online" for s in snaps))

        hw = DemoHardwareCollector().collect()
        self.assertTrue(hw.gpus and hw.gpus[0].mem_total > 0)

        diags = evaluate(hw, snaps)
        self.assertIsInstance(diags, list)
        self.assertGreater(len(diags), 0)  # 演示数据必然触发若干诊断


class TestHeadlessTui(unittest.TestCase):
    def test_app_compose_and_render(self):
        from llmscope.ui.app import LlmScopeApp
        from llmscope.ui.widgets import DiagnosticPanel, EngineTable, HardwareOverview

        async def _run():
            app = LlmScopeApp(demo=True)
            async with app.run_test(size=(110, 34)) as pilot:
                await pilot.pause()
                for _ in range(5):
                    await pilot.pause(0.25)
                self.assertIsNotNone(app.query_one(HardwareOverview))
                self.assertIsNotNone(app.query_one(EngineTable))
                self.assertIsNotNone(app.query_one(DiagnosticPanel))
                self.assertGreaterEqual(len(app._snaps), 3)
                svg = app.export_screenshot()
                self.assertIn("<svg", svg)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()

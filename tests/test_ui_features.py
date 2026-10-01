"""v0.2.0 新特性测试:GPU 短名、进程明细、详情页、无数据弱化占位。"""
import asyncio
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console


class TestShortGpuName(unittest.TestCase):
    def test_typical_names(self):
        from llmscope.utils import short_gpu_name

        cases = {
            "NVIDIA GeForce RTX 3050 Laptop GPU": "RTX 3050 Laptop",
            "NVIDIA GeForce RTX 4090": "RTX 4090",
            "NVIDIA GeForce GTX 1660 SUPER": "GTX 1660 SUPER",
            "NVIDIA A100-SXM4-40GB": "A100 SXM4 40GB",
            "Tesla V100-SXM2-16GB": "V100 SXM2 16GB",
            "NVIDIA H100 PCIe": "H100 PCIe",
        }
        for full, want in cases.items():
            self.assertEqual(short_gpu_name(full), want, full)


class TestHardwareProcessTable(unittest.TestCase):
    def test_render_with_processes(self):
        from llmscope.demo import DemoHardwareCollector
        from llmscope.ui.widgets import render_hardware

        hw = DemoHardwareCollector().collect()
        console = Console(record=True, width=130)
        console.print(render_hardware(hw, show_processes=True))
        text = console.export_text()
        self.assertIn("PID", text)
        self.assertIn("1234", text)          # demo 进程
        self.assertIn("GiB", text)           # 有显存数值


class TestEngineDetail(unittest.TestCase):
    def test_render_detail(self):
        from llmscope.demo import DemoEngineManager
        from llmscope.ui.widgets import render_engine_detail

        snap = DemoEngineManager().collect_all()[1]   # vllm
        console = Console(record=True, width=130)
        console.print(render_engine_detail(snap, show_raw=False))
        text = console.export_text()
        self.assertIn("吞吐与请求", text)
        self.assertIn("资源", text)

    def test_open_and_close_via_headless_app(self):
        from llmscope.ui.app import LlmScopeApp

        async def _run():
            app = LlmScopeApp(demo=True)
            async with app.run_test(size=(120, 36)) as pilot:
                await pilot.pause()
                for _ in range(4):
                    await pilot.pause(0.25)
                # 走真实按键路径:焦点在 DataTable 上,回车必须穿透其
                # 内置的"选中行"动作(priority binding)
                await pilot.press("enter")
                await pilot.pause()
                self.assertIn("EngineDetailScreen", type(app.screen).__name__)
                # 详情页打开时再按回车不应叠加新浮层
                await pilot.press("enter")
                await pilot.pause()
                self.assertIn("EngineDetailScreen", type(app.screen).__name__)
                await pilot.press("escape")
                await pilot.pause()
                self.assertNotIn("EngineDetailScreen", type(app.screen).__name__)

        asyncio.run(_run())


class TestDimPlaceholders(unittest.TestCase):
    def test_ollama_unsupported_cells_dim(self):
        from llmscope.models import EngineSnapshot
        from llmscope.ui.widgets import ENGINE_COLUMNS, engine_cells

        snap = EngineSnapshot(engine="ollama", port=11434, model="m:1b")
        cells = dict(zip(ENGINE_COLUMNS, engine_cells(snap)))
        for col in ("KV%", "队列", "运行"):
            cell = cells[col]
            self.assertEqual(cell.plain, "—", col)
            self.assertIn("dim", cell.style or "", col)

    def test_vllm_values_not_dim(self):
        from llmscope.models import EngineSnapshot
        from llmscope.ui.widgets import ENGINE_COLUMNS, engine_cells

        snap = EngineSnapshot(
            engine="vllm", port=8000, running_requests=2,
            queue_depth=1, kv_cache_usage=0.5,
        )
        cells = dict(zip(ENGINE_COLUMNS, engine_cells(snap)))
        self.assertEqual(cells["运行"].plain, "2")
        self.assertNotIn("dim", cells["运行"].style or "")


if __name__ == "__main__":
    unittest.main()

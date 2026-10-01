"""采集器单元测试:对 mock 引擎服务验证指纹与指标解析。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmscope.collectors.ollama import OllamaCollector
from llmscope.collectors.sglang import SglangCollector
from llmscope.collectors.vllm import VllmCollector
from llmscope.config import SETTINGS

from mock_engines import _serve, start_mock_ollama, start_mock_sglang, start_mock_vllm


class TestFingerprints(unittest.TestCase):
    def test_vllm_fingerprint(self):
        server, port = start_mock_vllm()
        try:
            snap = VllmCollector.fingerprint("127.0.0.1", port)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.engine, "vllm")
            self.assertEqual(snap.version, "0.6.6")
        finally:
            server.shutdown()

    def test_sglang_fingerprint(self):
        server, port = start_mock_sglang()
        try:
            snap = SglangCollector.fingerprint("127.0.0.1", port)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.engine, "sglang")
        finally:
            server.shutdown()

    def test_ollama_fingerprint(self):
        server, port = start_mock_ollama()
        try:
            snap = OllamaCollector.fingerprint("127.0.0.1", port)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.version, "0.5.7")
        finally:
            server.shutdown()

    def test_fingerprint_rejects_gui_mirror(self):
        """Ollama 桌面版 GUI:/api/version 是 JSON 但 /api/ps 回落成 HTML → 必须拒绝。"""
        gui = _serve(
            {
                "/api/version": lambda: _json(200, {"version": "0.34.2"}),
                "/api/ps": lambda: (200, b"<html>spa fallback</html>", "text/html"),
            }
        )
        try:
            self.assertIsNone(OllamaCollector.fingerprint("127.0.0.1", gui[1]))
        finally:
            gui[0].shutdown()

    def test_fingerprint_rejects_llamacpp(self):
        """llama.cpp(含 Ollama 内嵌引擎)的 /v1/models 同时带 OpenAI 形状 → 必须拒绝。"""
        llamacpp = _serve(
            {
                "/get_model_info": lambda: _json(
                    404, {"error": {"message": "File Not Found"}}
                ),
                "/api/version": lambda: (404, b"", "text/plain"),
                "/version": lambda: (404, b"", "text/plain"),
                "/metrics": lambda: (501, b"", "text/plain"),
                "/v1/models": lambda: _json(
                    200,
                    {
                        "models": [{"name": r"C:\models\qwen.gguf"}],
                        "object": "list",
                        "data": [{"id": r"C:\models\qwen.gguf"}],
                    },
                ),
            }
        )
        try:
            self.assertIsNone(VllmCollector.fingerprint("127.0.0.1", llamacpp[1]))
        finally:
            llamacpp[0].shutdown()

    def test_no_cross_match(self):
        """vLLM 指纹不得把 Ollama / SGLang 的端口认成自己。"""
        o_server, o_port = start_mock_ollama()
        s_server, s_port = start_mock_sglang()
        try:
            self.assertIsNone(VllmCollector.fingerprint("127.0.0.1", o_port))
            self.assertIsNone(VllmCollector.fingerprint("127.0.0.1", s_port))
        finally:
            o_server.shutdown()
            s_server.shutdown()


class TestCollect(unittest.TestCase):
    def test_vllm_collect(self):
        server, port = start_mock_vllm()
        try:
            col = VllmCollector()
            snap1 = col.collect("127.0.0.1", port)
            self.assertEqual(snap1.model, "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(snap1.running_requests, 2)
            self.assertEqual(snap1.queue_depth, 3)
            self.assertAlmostEqual(snap1.kv_cache_usage, 0.63)
            self.assertIsNone(snap1.gen_tokens_per_sec)  # 首次只有基线

            snap2 = col.collect("127.0.0.1", port, prev=snap1)
            self.assertIsNotNone(snap2.gen_tokens_per_sec)
            self.assertGreater(snap2.gen_tokens_per_sec, 0)
        finally:
            server.shutdown()

    def test_sglang_collect(self):
        server, port = start_mock_sglang()
        try:
            snap = SglangCollector().collect("127.0.0.1", port)
            self.assertEqual(snap.model, "Qwen2.5-7B-Instruct")
            self.assertEqual(snap.version, "0.3.6")
            self.assertAlmostEqual(snap.gen_tokens_per_sec, 123.4)
            self.assertAlmostEqual(snap.kv_cache_usage, 0.42)
        finally:
            server.shutdown()

    def test_ollama_collect(self):
        server, port = start_mock_ollama()
        old = SETTINGS.ollama_probe_enabled
        SETTINGS.ollama_probe_enabled = False  # 测试不真发探针
        try:
            snap = OllamaCollector().collect("127.0.0.1", port)
            self.assertEqual(snap.model, "qwen2.5:14b")
            self.assertEqual(snap.extra["installed"], ["qwen2.5:14b"])
            self.assertEqual(snap.vram_bytes, 10400000000)
            # 10.4G/11.6G → offload 约 10.3%
            self.assertIsNotNone(snap.cpu_offload_ratio)
            self.assertLess(snap.cpu_offload_ratio, 0.15)
            self.assertEqual(snap.extra["probe"], "disabled")
        finally:
            SETTINGS.ollama_probe_enabled = old
            server.shutdown()


if __name__ == "__main__":
    unittest.main()

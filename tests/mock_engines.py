"""伪造的三引擎 HTTP 服务,用于离线验证指纹与指标解析。

仅用于测试(ephemeral 端口),不会与真实引擎抢端口。
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Router(BaseHTTPRequestHandler):
    """按 {path: handler} 分发;handler 返回 (status, body_bytes, content_type)。"""

    routes: dict = {}

    def do_GET(self):  # noqa: N802 (http.server 命名约定)
        path = self.path.split("?")[0]
        handler = self.routes.get(path)
        if handler is None:
            self.send_response(404)
            self.end_headers()
            return
        status, body, ctype = handler()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静默访问日志
        pass


def _serve(routes: dict) -> tuple[ThreadingHTTPServer, int]:
    router = type("H", (_Router,), {"routes": routes})
    server = ThreadingHTTPServer(("127.0.0.1", 0), router)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _json(status: int, obj):
    import json

    return status, json.dumps(obj).encode(), "application/json"


def start_mock_vllm():
    """vLLM 假服务:版本 / 模型 / Prometheus 指标(token 计数器自增)。"""
    counter = {"gen": 1000.0, "prompt": 5000.0}

    def metrics():
        counter["gen"] += 260.0
        counter["prompt"] += 1500.0
        text = (
            '# HELP vllm:num_requests_running Number of running requests.\n'
            '# TYPE vllm:num_requests_running gauge\n'
            'vllm:num_requests_running{model_path="meta-llama/Llama-3.1-8B-Instruct"} 2.0\n'
            '# TYPE vllm:num_requests_waiting gauge\n'
            'vllm:num_requests_waiting{model_path="meta-llama/Llama-3.1-8B-Instruct"} 3.0\n'
            '# TYPE vllm:num_requests_preempted gauge\n'
            'vllm:num_requests_preempted 1.0\n'
            '# TYPE vllm:gpu_cache_usage_perc gauge\n'
            'vllm:gpu_cache_usage_perc{model_path="meta-llama/Llama-3.1-8B-Instruct"} 0.63\n'
            '# TYPE vllm:generation_tokens_total counter\n'
            f'vllm:generation_tokens_total {counter["gen"]}\n'
            '# TYPE vllm:prompt_tokens_total counter\n'
            f'vllm:prompt_tokens_total {counter["prompt"]}\n'
        )
        return 200, text.encode(), "text/plain"

    routes = {
        "/version": lambda: _json(200, {"version": "0.6.6"}),
        "/v1/models": lambda: _json(
            200, {"object": "list", "data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}
        ),
        # 故意暴露 SGLang 特征路径为 404,确认指纹不误判
        "/get_model_info": lambda: (404, b"", "text/plain"),
        "/metrics": metrics,
    }
    return _serve(routes)


def start_mock_sglang():
    routes = {
        "/get_model_info": lambda: _json(
            200, {"model_path": "Qwen/Qwen2.5-7B-Instruct"}
        ),
        "/get_server_info": lambda: _json(200, {"version": "0.3.6"}),
        "/metrics": lambda: (
            200,
            (
                '# TYPE sglang:num_running_reqs gauge\n'
                'sglang:num_running_reqs 1.0\n'
                '# TYPE sglang:num_queue_reqs gauge\n'
                'sglang:num_queue_reqs 0.0\n'
                '# TYPE sglang:num_retracted_reqs gauge\n'
                'sglang:num_retracted_reqs 0.0\n'
                '# TYPE sglang:token_usage gauge\n'
                'sglang:token_usage 0.42\n'
                '# TYPE sglang:gen_throughput gauge\n'
                'sglang:gen_throughput 123.4\n'
                '# TYPE sglang:prompt_throughput gauge\n'
                'sglang:prompt_throughput 890.0\n'
            ).encode(),
            "text/plain",
        ),
    }
    return _serve(routes)


def start_mock_ollama():
    routes = {
        "/api/version": lambda: _json(200, {"version": "0.5.7"}),
        "/api/tags": lambda: _json(
            200, {"models": [{"name": "qwen2.5:14b", "size": 11600000000}]}
        ),
        "/api/ps": lambda: _json(
            200,
            {
                "models": [
                    {
                        "name": "qwen2.5:14b",
                        "size": 11600000000,
                        "size_vram": 10400000000,
                        "details": {"quantization_level": "Q4_K_M"},
                    }
                ]
            },
        ),
    }
    return _serve(routes)

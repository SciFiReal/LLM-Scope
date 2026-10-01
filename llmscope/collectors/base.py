"""引擎采集器基类与 HTTP 小工具。

新增引擎只需三步:
1. 写一个 BaseEngineCollector 子类,实现 fingerprint / collect;
2. 加 @register 装饰器注册(注意在 collectors/__init__.py 中的导入顺序,
   它决定端口指纹判定的优先级);
3. 在 config.Settings.default_ports 里补默认端口。
发现、UI、诊断层会自动接入,无需改动其它模块。
"""
from __future__ import annotations

from typing import ClassVar, Optional

import requests

from ..config import SETTINGS
from ..models import EngineSnapshot

# 指纹探测超时:比常规请求稍宽松,但不能拖慢发现节奏
FP_TIMEOUT = SETTINGS.scan_connect_timeout + 1.0


def http_get(url: str, timeout: Optional[float] = None) -> requests.Response:
    return requests.get(url, timeout=timeout or SETTINGS.http_timeout)


def get_json(url: str, timeout: Optional[float] = None):
    """GET 并解析 JSON;任何失败返回 None(指纹探测要求"温和失败")。"""
    try:
        r = http_get(url, timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


class BaseEngineCollector:
    """引擎采集器基类。

    类属性(子类必须覆盖):
        engine_name      引擎标识,对应 models.OLLAMA 等常量
        default_ports    默认端口(发现阶段扫描)
        process_keywords 进程名 / 命令行匹配关键字(小写)

    实例方法 collect 可持有跨次采样的状态(如 counter 差分)。
    """

    engine_name: ClassVar[str] = "base"
    default_ports: ClassVar[tuple[int, ...]] = ()
    process_keywords: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def fingerprint(cls, host: str, port: int) -> Optional[EngineSnapshot]:
        """判断 host:port 是否运行本引擎。

        命中则返回携带基础信息(版本等)的快照;否则返回 None。
        实现要求:只发 1~2 个轻量请求,任何异常一律返回 None。
        """
        raise NotImplementedError

    def collect(
        self, host: str, port: int, prev: Optional[EngineSnapshot] = None
    ) -> EngineSnapshot:
        """采集完整指标。prev 是上一次快照,可用于继承信息 / 计算速率。"""
        raise NotImplementedError

    @staticmethod
    def _base(host: str, port: int) -> str:
        return f"http://{host}:{port}"

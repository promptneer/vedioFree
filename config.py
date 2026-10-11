"""配置模块：环境变量 + 命令行参数，集中管理运行配置。

AI 供应商通过 --provider 切换（deepseek / minimax / qwen / kimi / doubao），
均为 OpenAI 兼容接口，见 tensor_editor/providers.py。
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from tensor_editor.providers import PROVIDERS


@dataclass
class EngineConfig:
    """引擎运行配置。"""

    input: str = "./videos"
    output: str = "./output"
    template: str = "./template.json"
    workers: int = 4
    width: int = 1280
    height: int = 720
    fps: int = 30
    use_ai: bool = False
    provider: str = "deepseek"
    model: str = ""  # 空 → 用该 provider 的默认模型
    base_url: str = ""  # 空 → 用该 provider 的默认地址
    api_key: str = field(default="", repr=False)

    def resolved(self) -> "EngineConfig":
        """按 provider 注册表补全 model/base_url，并从环境变量读 Key。"""
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"未知 --provider: {self.provider}，可用: {sorted(PROVIDERS)}"
            )
        info = PROVIDERS[self.provider]
        if not self.model:
            self.model = info["model"] or ""
        if not self.base_url:
            self.base_url = info["base_url"] or ""
        if not self.api_key:
            self.api_key = os.environ.get(info["env_key"] or "", "")
        return self

    def require_api_key(self) -> None:
        """需要调用 AI 时检查 Key 是否存在（豆包同时检查 model）。"""
        info = PROVIDERS[self.provider]
        if not self.api_key:
            raise ValueError(
                f"未找到 {info['env_key']}：请 export "
                f"{info['env_key']}=你的Key 后再运行（{info['label']}）"
            )
        if not self.model:
            raise ValueError(
                f"{info['label']} 须用 --model 显式指定模型（{info['notes']}）"
            )


def parse_args(argv: list[str] | None = None) -> EngineConfig:
    """解析命令行参数，返回 EngineConfig。"""
    parser = argparse.ArgumentParser(
        description="AI 原生视频张量编辑引擎：一键批量处理"
    )
    parser.add_argument("--input", default="./videos", help="输入视频文件或目录")
    parser.add_argument("--output", default="./output", help="输出目录")
    parser.add_argument("--template", default="./template.json", help="模板 JSON/YAML 路径")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--width", type=int, default=1280, help="处理宽度")
    parser.add_argument("--height", type=int, default=720, help="处理高度")
    parser.add_argument("--fps", type=int, default=30, help="处理帧率")
    parser.add_argument(
        "--ai", action="store_true",
        help="启用 AI 指挥模板中的 ai_instruction（需对应 provider 的 API Key）",
    )
    parser.add_argument(
        "--provider", default="deepseek", choices=sorted(PROVIDERS),
        help="AI 供应商（均为 OpenAI 兼容接口），默认 deepseek",
    )
    parser.add_argument(
        "--model", default="",
        help="模型名，空则用 provider 默认；豆包必须显式给火山方舟推理接入点 ID",
    )
    parser.add_argument(
        "--base-url", default="", dest="base_url",
        help="覆盖 provider 默认 API 地址（默认空=不覆盖）",
    )
    args = parser.parse_args(argv)

    cfg = EngineConfig(
        input=args.input,
        output=args.output,
        template=args.template,
        workers=args.workers,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_ai=args.ai,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )
    return cfg.resolved()

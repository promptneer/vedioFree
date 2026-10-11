"""AI Provider 注册表：所有 OpenAI 兼容接口的模型供应商。

新增供应商只需在这里加一条（label / base_url / model / env_key），
无需改 Agent 逻辑。所有供应商都通过 OpenAI 兼容的
``POST /v1/chat/completions`` + Function Calling 驱动。
"""
from __future__ import annotations

from typing import Dict, Optional

# model 为 None 表示该供应商没有可用的默认模型，必须由用户显式传入
# （如豆包：model 须为火山方舟控制台创建的推理接入点 ID）。
PROVIDERS: Dict[str, Dict[str, Optional[str]]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "notes": "",
    },
    "minimax": {
        "label": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "model": "MiniMax-M2.7",
        "env_key": "MINIMAX_API_KEY",
        "notes": "国内站默认 api.minimax.chat；国际站用 --base-url 覆盖为 https://api.minimax.io/v1",
    },
    "qwen": {
        "label": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "notes": "",
    },
    "kimi": {
        "label": "Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2-0905-preview",
        "env_key": "MOONSHOT_API_KEY",
        "notes": "",
    },
    "doubao": {
        "label": "豆包（火山方舟）",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": None,
        "env_key": "ARK_API_KEY",
        "notes": "model 必须用 --model 传入火山方舟控制台创建的推理接入点 ID",
    },
}


def list_providers() -> Dict[str, str]:
    """返回 {provider名: 中文标签}，供 CLI --help / 文档使用。"""
    return {name: str(info["label"]) for name, info in PROVIDERS.items()}

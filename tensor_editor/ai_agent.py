"""AI 指挥模块：通过 Function Calling 指挥张量编辑引擎。

支持多个 OpenAI 兼容接口的供应商（DeepSeek / MiniMax / 通义 / Kimi / 豆包），
见 tensor_editor/providers.py。用 --provider 切换。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .providers import PROVIDERS

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    from openai import APITimeoutError, AuthenticationError, RateLimitError, APIError
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

    class _DummyError(Exception):
        """openai 未安装时的占位异常，保证 except 子句语法合法。"""

    APITimeoutError = AuthenticationError = RateLimitError = APIError = _DummyError


class AgentError(Exception):
    """Agent 运行失败时抛出。"""


def _tool(name: str, description: str,
          properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _int_prop(desc: str) -> Dict[str, Any]:
    return {"type": "integer", "description": desc}


def _num_prop(desc: str) -> Dict[str, Any]:
    return {"type": "number", "description": desc}


def _str_prop(desc: str) -> Dict[str, Any]:
    return {"type": "string", "description": desc}


TOOLS: List[Dict[str, Any]] = [
    _tool(
        "cut_frames", "从指定视频切出 [start, end) 帧，生成一个新视频并返回其 ID。",
        {
            "video_id": _str_prop("已注册的视频 ID"),
            "start": _int_prop("起始帧（含），从 0 开始"),
            "end": _int_prop("结束帧（不含）"),
        },
        ["video_id", "start", "end"],
    ),
    _tool(
        "modify_region", "修改某一帧的矩形区域像素（原地）。",
        {
            "video_id": _str_prop("已注册的视频 ID"),
            "frame_idx": _int_prop("帧下标"),
            "y1": _int_prop("行起点（含）"),
            "y2": _int_prop("行终点（不含）"),
            "x1": _int_prop("列起点（含）"),
            "x2": _int_prop("列终点（不含）"),
            "operation": _str_prop(
                "操作：invert（反色）/black/white/red/green/blue/yellow，"
                "或 R,G,B 三元组如 '255,0,0'"
            ),
        },
        ["video_id", "frame_idx", "y1", "y2", "x1", "x2", "operation"],
    ),
    _tool(
        "concatenate_videos", "把两个视频在时间维拼接，生成新视频并返回其 ID。",
        {
            "video_a_id": _str_prop("第一个视频 ID"),
            "video_b_id": _str_prop("第二个视频 ID"),
            "new_id": _str_prop("新视频 ID（可选，不给则自动生成）"),
        },
        ["video_a_id", "video_b_id"],
    ),
    _tool(
        "apply_color", "整体调色（原地）。",
        {
            "video_id": _str_prop("已注册的视频 ID"),
            "brightness": _num_prop("亮度倍数，1.0 为不变"),
            "contrast": _num_prop("对比度倍数，1.0 为不变"),
            "saturation": _num_prop("饱和度倍数，1.0 为不变"),
        },
        ["video_id"],
    ),
    _tool(
        "apply_transition", "加转场（原地）：fadein 从黑场淡入，fadeout 淡出到黑场，fade 两者都有。",
        {
            "video_id": _str_prop("已注册的视频 ID"),
            "type": _str_prop("转场类型：fadein/fadeout/fade"),
            "duration": _int_prop("转场帧数"),
        },
        ["video_id"],
    ),
    _tool(
        "export_video", "把指定视频编码导出为 MP4 文件。",
        {
            "video_id": _str_prop("已注册的视频 ID"),
            "output_path": _str_prop("输出 MP4 路径"),
            "fps": _int_prop("输出帧率，默认 30"),
        },
        ["video_id", "output_path"],
    ),
]


class VideoAIAgent:
    """AI 指挥官：把自然语言指令翻译成张量操作并执行。

    支持 providers.py 注册的所有 OpenAI 兼容供应商，
    默认 DeepSeek。

    Example:
        >>> agent = VideoAIAgent(provider="qwen")  # 通义千问
        >>> agent.register_video("main", TensorEditor(frames))
        >>> agent.run("把前 50 帧切出来，调暖一点，导出到 out.mp4", video_id="main")
    """

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        """初始化。

        Args:
            provider: 供应商名，见 providers.PROVIDERS
               （deepseek / minimax / qwen / kimi / doubao）。
            api_key: 该供应商的 API Key；不给则读其对应的环境变量。
            model: 模型名；不给则用该供应商的默认模型。
                豆包无默认模型，必须显式传入火山方舟的推理接入点 ID。
            base_url: API 地址；不给则用该供应商的默认值。
            client: 可注入自定义 OpenAI 兼容客户端（便于测试打桩）。
        """
        if provider not in PROVIDERS:
            raise AgentError(
                f"未知 provider: {provider}，可用: {sorted(PROVIDERS)}"
            )
        info = PROVIDERS[provider]
        self.provider = provider
        self.provider_label = str(info["label"])
        self.base_url = base_url or str(info["base_url"])
        self.model = model or info["model"]
        if not self.model:
            raise AgentError(
                f"{self.provider_label} 没有默认模型：请通过 model 参数 / "
                f"--model 显式指定（{info['notes']}）"
            )
        if client is not None:
            self.client = client
        else:
            if not _OPENAI_AVAILABLE:
                raise AgentError("需要 openai 包：pip install openai")
            key = api_key or os.environ.get(str(info["env_key"]))
            if not key:
                raise AgentError(
                    f"未找到 {info['env_key']}：请设置环境变量 "
                    f"{info['env_key']}=你的Key 后再运行"
                )
            self.client = OpenAI(api_key=key, base_url=self.base_url)
        self.videos: Dict[str, Any] = {}
        self.messages: List[Dict[str, Any]] = []
        self._cut_counter = 0
        logger.info("AI 指挥官初始化: %s / %s", self.provider_label, self.model)

    # ---------- 视频注册表 ----------

    def register_video(self, video_id: str, editor: Any) -> None:
        """注册一个 TensorEditor，供 tool 调用。"""
        self.videos[video_id] = editor
        logger.info("注册视频: %s shape=%s", video_id, editor.shape)

    def get_video(self, video_id: str) -> Any:
        """取已注册的视频，未注册抛 AgentError。"""
        try:
            return self.videos[video_id]
        except KeyError as exc:
            raise AgentError(
                f"未知 video_id: {video_id}，已注册: {sorted(self.videos)}"
            ) from exc

    def _meta_info(self) -> str:
        lines = ["当前已注册的视频（张量元信息）："]
        for vid, ed in self.videos.items():
            t, h, w, c = ed.shape
            lines.append(
                f"- {vid}: T={t}帧 H={h} W={w} C={c} dtype={ed.frames.dtype}"
            )
        lines.append(
            "你是视频编辑指挥官，只能通过 tools 操作视频，"
            "不要输出多余解释；完成所有操作后用 export_video 导出。"
        )
        return "\n".join(lines)

    # ---------- Agent 主循环 ----------

    def run(self, instruction: str, max_rounds: int = 10) -> Optional[str]:
        """执行一条自然语言指令，返回 AI 的最终回复文本。

        Args:
            instruction: 如“去掉右上角水印，整体调暖色调，导出 out.mp4”。
            max_rounds: 最多 tool 调用轮数，防止死循环。
        """
        self.messages.append(
            {"role": "user", "content": f"{instruction}\n\n{self._meta_info()}"}
        )
        for round_no in range(1, max_rounds + 1):
            logger.info("Agent 第 %d 轮：请求 %s 决策", round_no, self.provider_label)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    timeout=60,
                )
            except APITimeoutError as exc:
                raise AgentError(f"{self.provider_label} API 超时（60s），请重试") from exc
            except AuthenticationError as exc:
                raise AgentError(
                    f"{self.provider_label} 鉴权失败：检查 API Key 是否正确、额度是否充足"
                ) from exc
            except RateLimitError as exc:
                raise AgentError(f"{self.provider_label} 限流（429），稍后再试") from exc
            except APIError as exc:
                raise AgentError(f"{self.provider_label} API 错误: {exc}") from exc

            msg = resp.choices[0].message
            # 兼容 SDK 对象与纯 dict 两种返回形态
            dump = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)
            self.messages.append(dump)

            tool_calls = dump.get("tool_calls") or []
            if not tool_calls:
                logger.info("Agent 完成，无更多 tool_calls")
                return dump.get("content")

            for tc in tool_calls:
                fn = tc["function"]
                name, raw_args = fn["name"], fn["arguments"]
                logger.info("AI 指令: %s(%s)", name, raw_args)
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError as exc:
                    result: Dict[str, Any] = {"ok": False, "error": f"参数不是合法 JSON: {exc}"}
                else:
                    result = self._execute_tool(name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
        raise AgentError(f"达到最大轮数（{max_rounds}），指令未执行完")

    # ---------- tool 执行分发 ----------

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "cut_frames":
                editor = self.get_video(args["video_id"])
                new_editor = editor.cut_frames(int(args["start"]), int(args["end"]))
                self._cut_counter += 1
                new_id = f"{args['video_id']}_cut{self._cut_counter}"
                self.register_video(new_id, new_editor)
                return {"ok": True, "video_id": new_id, "shape": new_editor.shape}

            if name == "modify_region":
                editor = self.get_video(args["video_id"])
                op = str(args.get("operation", "black"))
                if "," in op:  # "255,0,0" -> RGB 元组
                    op = tuple(int(v) for v in op.split(","))  # type: ignore[assignment]
                editor.modify_region(
                    int(args["frame_idx"]), int(args["y1"]), int(args["y2"]),
                    int(args["x1"]), int(args["x2"]), op,
                )
                return {"ok": True, "video_id": args["video_id"]}

            if name == "concatenate_videos":
                a = self.get_video(args["video_a_id"])
                b = self.get_video(args["video_b_id"])
                new_id = args.get("new_id") or f"{args['video_a_id']}_cat_{args['video_b_id']}"
                merged = a.concatenate(b)
                self.register_video(new_id, merged)
                return {"ok": True, "video_id": new_id, "shape": merged.shape}

            if name == "apply_color":
                editor = self.get_video(args["video_id"])
                editor.apply_color(
                    float(args.get("brightness", 1.0)),
                    float(args.get("contrast", 1.0)),
                    float(args.get("saturation", 1.0)),
                )
                return {"ok": True, "video_id": args["video_id"]}

            if name == "apply_transition":
                editor = self.get_video(args["video_id"])
                editor.apply_transition(
                    str(args.get("type", "fade")), int(args.get("duration", 10)))
                return {"ok": True, "video_id": args["video_id"]}

            if name == "export_video":
                from .encode import encode_tensor_to_video

                editor = self.get_video(args["video_id"])
                out = str(args["output_path"])
                encode_tensor_to_video(editor.frames, out, fps=int(args.get("fps", 30)))
                return {"ok": True, "output_path": out}

            return {"ok": False, "error": f"未知工具: {name}"}
        except (KeyError, ValueError, TypeError) as exc:
            return {"ok": False, "error": f"参数错误: {exc}"}
        except Exception as exc:  # 编辑/编码异常也如实回传
            logger.exception("工具 %s 执行失败", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def reset_history(self) -> None:
        """清空多轮对话上下文（视频注册表保留）。"""
        self.messages = []

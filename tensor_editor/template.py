"""模板系统：JSON/YAML 描述的编辑流程，支持继承、覆盖与批量套用。"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TemplateError(Exception):
    """模板加载或应用失败时抛出。"""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深合并：override 的值覆盖 base，字典递归合并。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class VideoTemplate:
    """视频编辑模板。

    配置示例见 template.json。``apply`` 按顺序执行确定性操作；
    ``ai_instruction`` 不在这里直接调用 API，而是返回给调用方
    决定是否走 VideoAIAgent。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TemplateError("模板配置须为 dict")
        self.config = config

    # ---------- 加载 ----------

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VideoTemplate":
        return cls(config)

    @classmethod
    def from_json(cls, path: str) -> "VideoTemplate":
        if not os.path.isfile(path):
            raise TemplateError(f"模板文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls._with_inheritance(config, os.path.dirname(os.path.abspath(path)))

    @classmethod
    def from_yaml(cls, path: str) -> "VideoTemplate":
        try:
            import yaml
        except ImportError as exc:
            raise TemplateError("YAML 模板需要 pyyaml：pip install pyyaml") from exc
        if not os.path.isfile(path):
            raise TemplateError(f"模板文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls._with_inheritance(config, os.path.dirname(os.path.abspath(path)))

    @classmethod
    def _with_inheritance(cls, config: Dict[str, Any], base_dir: str) -> "VideoTemplate":
        parent = config.get("extends")
        if parent:
            parent_path = parent if os.path.isabs(parent) else os.path.join(base_dir, parent)
            if parent_path.endswith((".yaml", ".yml")):
                parent_tpl = cls.from_yaml(parent_path)
            else:
                parent_tpl = cls.from_json(parent_path)
            config = _deep_merge(parent_tpl.config, config)
        return cls(config)

    # ---------- 应用 ----------

    @property
    def ai_instruction(self) -> Optional[str]:
        """模板里的 AI 自然语言指令（调用方决定是否送给 VideoAIAgent）。"""
        return self.config.get("ai_instruction")

    def apply(
        self,
        editor: Any,
        assets: Optional[Dict[str, Any]] = None,
        decode_fn: Any = None,
    ) -> Any:
        """按模板顺序对编辑器执行确定性操作，返回编辑器（链式）。

        顺序：segments 切段拼接 -> intro/outro 片头片尾 ->
        color 调色 -> transition 转场。

        Args:
            editor: TensorEditor。
            assets: {"intro": TensorEditor, "outro": TensorEditor} 或路径；
                为路径时需要 decode_fn。
            decode_fn: 可选，签名 ``(path) -> TensorEditor``，用于解码片头片尾。
        """
        cfg = self.config

        segments: Optional[List[Dict[str, int]]] = cfg.get("segments")
        if segments:
            parts = [editor.cut_frames(s["start"], s["end"]) for s in segments]
            merged = parts[0]
            for part in parts[1:]:
                merged = merged.concatenate(part)
            editor = merged
            logger.info("模板: 切段拼接 %d 段", len(parts))

        intro = self._resolve_asset(cfg.get("intro"), assets, decode_fn, "intro")
        outro = self._resolve_asset(cfg.get("outro"), assets, decode_fn, "outro")
        if intro is not None:
            editor = intro.concatenate(editor)
            logger.info("模板: 拼接片头")
        if outro is not None:
            editor = editor.concatenate(outro)
            logger.info("模板: 拼接片尾")

        color: Optional[Dict[str, float]] = cfg.get("color")
        if color:
            editor.apply_color(
                brightness=float(color.get("brightness", 1.0)),
                contrast=float(color.get("contrast", 1.0)),
                saturation=float(color.get("saturation", 1.0)),
            )
            logger.info("模板: 调色 %s", color)

        transition: Optional[str] = cfg.get("transition")  # type: ignore[assignment]
        if transition:
            duration = int(cfg.get("transition_duration", 10))
            editor.apply_transition(transition, duration)
            logger.info("模板: 转场 %s(%d帧)", transition, duration)

        return editor

    @staticmethod
    def _resolve_asset(
        value: Any,
        assets: Optional[Dict[str, Any]],
        decode_fn: Any,
        name: str,
    ) -> Any:
        if value is None:
            return None
        if assets and name in assets:
            return assets[name]
        if isinstance(value, str):
            if decode_fn is None:
                raise TemplateError(
                    f"模板引用了 {name} 路径 {value}，但未提供 decode_fn 解码"
                )
            return decode_fn(value)
        raise TemplateError(f"{name} 配置非法: {value!r}")

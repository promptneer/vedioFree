"""入口：一键运行批量视频张量编辑。

示例：
    python main.py --input ./videos --output ./output --template ./template.json
    # 通义千问指挥：
    export DASHSCOPE_API_KEY=sk-xxx
    python main.py --input ./videos --output ./output --template ./template.json \
        --ai --provider qwen
    # 豆包（须填火山方舟推理接入点 ID）：
    export ARK_API_KEY=xxx
    python main.py --input ./videos --output ./output --template ./template.json \
        --ai --provider doubao --model ep-xxxxxxxx
"""
from __future__ import annotations

import glob
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("main")

from config import parse_args
from tensor_editor import TensorEditor  # noqa: F401  (供模板/测试引用)
from tensor_editor.ai_agent import VideoAIAgent
from tensor_editor.batch import batch_process
from tensor_editor.template import VideoTemplate

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def collect_videos(input_path: str) -> list[str]:
    """输入为文件则单条，为目录则收集常见视频格式。"""
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        videos.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
    return sorted(videos)


def main() -> int:
    cfg = parse_args()
    videos = collect_videos(cfg.input)
    if not videos:
        logger.error("没有找到视频文件: %s", cfg.input)
        return 1
    logger.info("待处理 %d 条视频", len(videos))

    template = (
        VideoTemplate.from_yaml(cfg.template)
        if cfg.template.endswith((".yaml", ".yml"))
        else VideoTemplate.from_json(cfg.template)
    )

    agent = None
    if cfg.use_ai:
        cfg.require_api_key()
        agent = VideoAIAgent(
            provider=cfg.provider,
            api_key=cfg.api_key,
            model=cfg.model or None,
            base_url=cfg.base_url or None,
        )
        logger.info("AI 指挥已启用: %s / %s", cfg.provider, agent.model)
    elif template.ai_instruction:
        logger.warning("模板含 ai_instruction，但未加 --ai 参数，将跳过 AI 指挥")

    results = batch_process(
        videos,
        template,
        cfg.output,
        max_workers=cfg.workers,
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        ai_agent=agent,
    )
    failed = [(v, e) for v, o, e in results if o is None]
    for v, e in failed:
        logger.error("失败: %s -> %s", v, e)
    logger.info("完成：%d 成功 / %d 总数", len(results) - len(failed), len(results))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())

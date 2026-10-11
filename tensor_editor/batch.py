"""批量处理：多线程并发对多条视频套用同一模板。"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

Result = Tuple[str, str | None, str | None]  # (输入, 输出|None, 错误|None)


def batch_process(
    video_list: List[str],
    template: Any,
    output_dir: str,
    max_workers: int = 4,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    max_retries: int = 2,
    show_progress: bool = True,
    ai_agent: Any = None,
) -> List[Result]:
    """批量处理视频。

    Args:
        video_list: 输入视频路径列表。
        template: VideoTemplate / dict / 模板文件路径。
        output_dir: 输出目录（自动创建）。
        max_workers: 并发线程数。
        width, height, fps: 解码与输出规格。
        max_retries: 单条失败重试次数。
        show_progress: 是否显示 tqdm 进度条。
        ai_agent: 可选 VideoAIAgent；模板含 ai_instruction 时逐条调用。
            注意：AI 指挥步骤在批量中串行执行（保证每条视频对话上下文干净），
            解码/编码仍并发。

    Returns:
        每条 (输入路径, 输出路径或 None, 错误信息或 None)。
    """
    from .decode import decode_video_to_tensor
    from .encode import encode_tensor_to_video
    from .edit_engine import TensorEditor
    from .template import VideoTemplate

    if isinstance(template, str):
        template = (
            VideoTemplate.from_yaml(template)
            if template.endswith((".yaml", ".yml"))
            else VideoTemplate.from_json(template)
        )
    elif isinstance(template, dict):
        template = VideoTemplate.from_dict(template)
    if not isinstance(template, VideoTemplate):
        raise ValueError("template 须为 VideoTemplate / dict / 文件路径")

    os.makedirs(output_dir, exist_ok=True)
    use_ai = ai_agent is not None and template.ai_instruction
    # AI 指挥串行化：VideoAIAgent 的对话上下文是按视频独立的，
    # 多线程同时 run 会串话；用锁保证一次只跑一条视频的 AI 流程，
    # 每条开始前 reset_history() 清空上一条的上下文。
    # 解码/编码仍保持并发。
    ai_lock = threading.Lock()

    def _decode_asset(path: str) -> TensorEditor:
        frames = decode_video_to_tensor(path, width, height, fps=fps)
        return TensorEditor(frames)

    def _process_one(video_path: str) -> Result:
        name = os.path.basename(video_path)
        stem, _ = os.path.splitext(name)
        out_path = os.path.join(output_dir, f"output_{stem}.mp4")
        last_err = "未知错误"
        for attempt in range(max_retries + 1):
            try:
                frames = decode_video_to_tensor(video_path, width, height, fps=fps)
                editor = TensorEditor(frames)
                editor = template.apply(editor, decode_fn=_decode_asset)
                if use_ai:
                    with ai_lock:
                        ai_agent.reset_history()
                        vid = f"batch_{stem}"
                        ai_agent.register_video(vid, editor)
                        ai_agent.run(template.ai_instruction or "", max_rounds=8)
                        editor = ai_agent.get_video(vid)
                encode_tensor_to_video(editor.frames, out_path, fps=fps)
                logger.info("完成: %s -> %s", video_path, out_path)
                return (video_path, out_path, None)
            except Exception as exc:  # noqa: BLE001 - 批量需兜底并记录
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning("失败(第%d次): %s -> %s", attempt + 1, video_path, last_err)
                time.sleep(1)
        logger.error("放弃: %s (%s)", video_path, last_err)
        return (video_path, None, last_err)

    results: List[Result] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, v): v for v in video_list}
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc="批量处理")  # type: ignore[assignment]
            except ImportError:
                logger.warning("未安装 tqdm，跳过进度条")
        for fut in iterator:
            results.append(fut.result())

    ok = sum(1 for _, out, _ in results if out)
    logger.info("批量完成: %d/%d 成功", ok, len(results))
    return results

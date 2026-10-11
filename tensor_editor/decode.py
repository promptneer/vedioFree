"""解码模块：FFmpeg rawvideo 管道 -> PyTorch 张量，全程不落盘。"""
from __future__ import annotations

import logging
import os
import subprocess

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DecodeError(Exception):
    """解码失败时抛出。"""


def decode_video_to_tensor(
    video_path: str,
    width: int,
    height: int,
    fps: int = 30,
) -> torch.Tensor:
    """把视频解码为形状 [T, H, W, C] 的 uint8 张量。

    使用 FFmpeg 的 ``rawvideo`` 管道输出 RGB24 逐帧读取，
    ``np.frombuffer`` 建数组后复制一份可写内存，
    再 ``torch.as_tensor`` 共享（之后不再拷贝）。

    Args:
        video_path: 输入视频路径。
        width: 解码宽度（FFmpeg 会缩放）。
        height: 解码高度（FFmpeg 会缩放）。
        fps: 解码帧率。

    Returns:
        形状为 [T, H, W, C]、dtype 为 uint8 的张量，C=3（RGB）。

    Raises:
        DecodeError: 文件不存在、参数非法、FFmpeg 缺失或解码失败。
    """
    if not os.path.isfile(video_path):
        raise DecodeError(f"文件不存在: {video_path}")
    if width <= 0 or height <= 0:
        raise DecodeError(f"分辨率非法: width={width}, height={height}")
    if fps <= 0:
        raise DecodeError(f"帧率非法: fps={fps}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise DecodeError("找不到 ffmpeg，可执行文件未安装或不在 PATH 中") from exc

    assert proc.stdout is not None
    raw: bytes = proc.stdout.read()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode(errors="ignore").strip()
        raise DecodeError(f"FFmpeg 解码失败（code={proc.returncode}）: {msg}")

    frame_bytes = width * height * 3
    if len(raw) == 0:
        raise DecodeError("解码得到 0 字节：文件可能损坏或格式不受支持")
    if len(raw) % frame_bytes != 0:
        logger.warning("数据长度不是整帧（%d 字节），截断多余字节", len(raw))
        raw = raw[: len(raw) // frame_bytes * frame_bytes]

    n_frames = len(raw) // frame_bytes
    # frombuffer 读 bytes 得到的是只读数组；copy 一份可写内存，
    # 否则后续原地编辑（modify_region 等）会写坏只读内存。
    # as_tensor 与这份可写数组共享内存，不再额外拷贝。
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, height, width, 3).copy()
    tensor = torch.as_tensor(arr)
    logger.info("解码完成: %s -> shape=%s", video_path, tuple(tensor.shape))
    return tensor

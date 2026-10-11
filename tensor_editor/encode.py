"""编码模块：PyTorch 张量 -> FFmpeg rawvideo 管道 -> H.264 MP4，全程不落盘。"""
from __future__ import annotations

import logging
import os
import subprocess

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EncodeError(Exception):
    """编码失败时抛出。"""


def encode_tensor_to_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 30,
    bitrate: str | None = None,
    crf: int = 18,
) -> None:
    """把 [T, H, W, C] uint8 张量编码为 H.264 MP4。

    通过 FFmpeg 的 ``rawvideo`` 管道逐帧写入 stdin，不经过中间文件。

    Args:
        frames: 形状 [T, H, W, C]、dtype uint8 的张量（C=3）。
            也接受 numpy 数组，会先转为张量。
        output_path: 输出 MP4 路径（父目录需存在）。
        fps: 输出帧率。
        bitrate: 可选码率，如 "2M"；不指定则用 crf 控制质量。
        crf: 恒定质量系数（0-51，越小越好），默认 18。

    Raises:
        EncodeError: 张量形状非法、FFmpeg 缺失或编码失败。
    """
    if isinstance(frames, np.ndarray):
        frames = torch.as_tensor(frames)
    if not isinstance(frames, torch.Tensor):
        raise EncodeError(f"frames 须为 torch.Tensor 或 np.ndarray，实际 {type(frames)}")
    if frames.ndim != 4:
        raise EncodeError(f"frames 须为 4 维 [T,H,W,C]，实际 {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise EncodeError(f"frames dtype 须为 uint8，实际 {frames.dtype}")
    t, h, w, c = frames.shape
    if t == 0:
        raise EncodeError("frames 为空（T=0），无法编码")
    if c != 3:
        raise EncodeError(f"仅支持 RGB（C=3），实际 C={c}")
    if fps <= 0:
        raise EncodeError(f"帧率非法: fps={fps}")

    parent = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(parent):
        raise EncodeError(f"输出目录不存在: {parent}")

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
    ]
    if bitrate:
        cmd += ["-b:v", bitrate]
    else:
        cmd += ["-crf", str(crf)]
    cmd.append(output_path)

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise EncodeError("找不到 ffmpeg，可执行文件未安装或不在 PATH 中") from exc

    assert proc.stdin is not None and proc.stderr is not None
    try:
        # 非连续内存（如切片/permute 后）先整理为连续再写
        arr = frames.detach().cpu().numpy()
        for i in range(t):
            frame = np.ascontiguousarray(arr[i])
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore").strip()
        proc.wait()
        raise EncodeError(f"写入编码管道中断，FFmpeg 说: {err}") from exc
    except Exception:
        try:
            proc.kill()
        finally:
            proc.wait()
        raise

    # 先关 stdin 让 FFmpeg 收尾，再读 stderr、等进程退出。
    # （communicate() 会在已关闭的 stdin 上 flush 而报错，故不用它。）
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="ignore").strip()
    returncode = proc.wait()
    if returncode != 0:
        raise EncodeError(f"FFmpeg 编码失败（code={returncode}）: {err}")
    logger.info("编码完成: shape=%s -> %s", tuple(frames.shape), output_path)

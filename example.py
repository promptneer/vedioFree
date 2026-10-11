"""最小可运行示例：解码 -> 切10帧 -> 改一块像素 -> 拼回去 -> 编码。

运行：python example.py
输出：example_out.mp4（可直接播放）
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from tensor_editor import decode_video_to_tensor, encode_tensor_to_video, TensorEditor

SAMPLE = "sample.mp4"
OUTPUT = "example_out.mp4"
W, H, FPS = 320, 240, 30


def make_sample_video(path: str = SAMPLE) -> None:
    """用 FFmpeg testsrc 生成一段 3 秒测试视频（如果不存在）。"""
    if os.path.isfile(path):
        return
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate={FPS}:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True)
    print(f"生成测试视频: {path}")


def main() -> None:
    make_sample_video()

    # 1. 解码为张量 [T, H, W, C]
    frames = decode_video_to_tensor(SAMPLE, W, H, fps=FPS)
    print(f"解码: shape={tuple(frames.shape)}, dtype={frames.dtype}")
    editor = TensorEditor(frames)

    # 2. 切出前 10 帧
    head = editor.cut_frames(0, 10)
    print(f"切帧: head={head.shape}")

    # 3. 在第 5 帧画一块红色矩形（改像素，原地）
    head.modify_region(frame_idx=5, y1=60, y2=120, x1=80, x2=200, operation="red")
    print("改像素: frame[5] 矩形区域 -> red")

    # 4. 拼回去：head(改过) + tail(剩下)
    tail = editor.cut_frames(10, editor.num_frames)
    out = head.concatenate(tail)
    print(f"拼接: out={out.shape}")

    # 5. 编码写出
    encode_tensor_to_video(out.frames, OUTPUT, fps=FPS)
    print(f"完成: {OUTPUT}，用播放器打开验证")


if __name__ == "__main__":
    sys.exit(main())

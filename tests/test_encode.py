"""编码模块测试：张量 -> MP4 -> ffprobe 校验。"""
import os
import subprocess

import pytest
import torch

from tensor_editor import encode_tensor_to_video, EncodeError, decode_video_to_tensor

OUT = "/tmp/tve_test_out.mp4"


def _frames(t=10, h=64, w=96):
    base = torch.arange(t * h * w * 3, dtype=torch.int64) % 256
    return base.to(torch.uint8).reshape(t, h, w, 3)


def test_encode_decode_roundtrip():
    frames = _frames()
    encode_tensor_to_video(frames, OUT, fps=15)
    assert os.path.isfile(OUT) and os.path.getsize(OUT) > 0
    back = decode_video_to_tensor(OUT, 96, 64, fps=15)
    assert back.shape == frames.shape
    # H.264 有损压缩：只要求大致接近
    diff = (back.float() - frames.float()).abs().mean().item()
    assert diff < 12, f"roundtrip 误差过大: {diff}"


def test_encode_bad_input():
    with pytest.raises(EncodeError):
        encode_tensor_to_video(torch.zeros(0, 64, 96, 3, dtype=torch.uint8), "/tmp/x.mp4")
    with pytest.raises(EncodeError):
        encode_tensor_to_video(torch.zeros(10, 64, 96), "/tmp/x.mp4")
    with pytest.raises(EncodeError):
        encode_tensor_to_video(torch.zeros(10, 64, 96, 3), "/tmp/x.mp4")  # float32


def test_encode_missing_dir():
    with pytest.raises(EncodeError):
        encode_tensor_to_video(_frames(), "/tmp/不存在的目录/x.mp4")


def test_output_is_playable():
    frames = _frames(t=15)
    encode_tensor_to_video(frames, OUT, fps=15)
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
         "-of", "default=noprint_wrappers=1", OUT],
        capture_output=True, text=True, check=True,
    )
    assert "codec_name=h264" in probe.stdout
    assert "width=96" in probe.stdout and "height=64" in probe.stdout

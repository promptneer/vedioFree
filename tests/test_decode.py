"""解码模块测试：需要 FFmpeg 与 testsrc（无需真实视频文件）。"""
import os
import subprocess

import pytest
import torch

from tensor_editor import decode_video_to_tensor, DecodeError

SAMPLE = "/tmp/tve_test_src.mp4"


@pytest.fixture(scope="module")
def sample_video():
    if not os.path.isfile(SAMPLE):
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=160x120:rate=15:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", SAMPLE],
            check=True,
        )
    return SAMPLE


def test_decode_shape_dtype(sample_video):
    t = decode_video_to_tensor(sample_video, 160, 120, fps=15)
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.uint8
    assert t.ndim == 4 and t.shape[1:] == (120, 160, 3)
    assert t.shape[0] == 30  # 2 秒 x 15fps


def test_decode_resize(sample_video):
    t = decode_video_to_tensor(sample_video, 80, 60, fps=15)
    assert t.shape[1:] == (60, 80, 3)


def test_decode_missing_file():
    with pytest.raises(DecodeError):
        decode_video_to_tensor("/tmp/不存在的视频.mp4", 160, 120)


def test_decode_bad_params(sample_video):
    with pytest.raises(DecodeError):
        decode_video_to_tensor(sample_video, 0, 120)
    with pytest.raises(DecodeError):
        decode_video_to_tensor(sample_video, 160, 120, fps=0)

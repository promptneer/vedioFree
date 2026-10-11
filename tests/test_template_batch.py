"""模板与批量测试：模板用纯张量，批量用 FFmpeg 生成的微型视频。"""
import json
import os
import subprocess

import pytest
import torch

from tensor_editor import TensorEditor
from tensor_editor.template import VideoTemplate

TMP = "/tmp/tve_tpl"


def _editor(t=30, value=100):
    return TensorEditor(torch.full((t, 32, 48, 3), value, dtype=torch.uint8))


def test_template_segments_and_color():
    tpl = VideoTemplate.from_dict({
        "segments": [{"start": 0, "end": 10}, {"start": 20, "end": 30}],
        "color": {"brightness": 2.0},
    })
    out = tpl.apply(_editor())
    assert out.num_frames == 20
    assert out.frames.max().item() == 200
    assert tpl.ai_instruction is None


def test_template_inheritance(tmp_path):
    base = tmp_path / "base.json"
    child = tmp_path / "child.json"
    base.write_text(json.dumps({"color": {"brightness": 1.0, "contrast": 1.5},
                                "transition": "fade"}), encoding="utf-8")
    child.write_text(json.dumps({"extends": str(base),
                                 "color": {"brightness": 2.0}}), encoding="utf-8")
    tpl = VideoTemplate.from_json(str(child))
    assert tpl.config["color"] == {"brightness": 2.0, "contrast": 1.5}
    assert tpl.config["transition"] == "fade"


def test_template_bad_file():
    with pytest.raises(Exception):
        VideoTemplate.from_json("/tmp/不存在的模板.json")


def _make_video(path, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # seed 只用来区分两个源：偶数用 testsrc，奇数用纯色源
    if seed % 2 == 0:
        src = "testsrc=size=64x48:rate=10:duration=1"
    else:
        src = "color=c=red:size=64x48:rate=10:duration=1"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True,
    )


def test_batch_process(tmp_path):
    from tensor_editor.batch import batch_process

    srcdir = str(tmp_path / "in")
    outdir = str(tmp_path / "out")
    vids = [os.path.join(srcdir, f"v{i}.mp4") for i in range(2)]
    for i, v in enumerate(vids):
        _make_video(v, seed=i)

    tpl = VideoTemplate.from_dict({
        "color": {"brightness": 1.2},
        "transition": "fadein",
        "transition_duration": 3,
    })
    results = batch_process(vids, tpl, outdir, max_workers=2,
                            width=64, height=48, fps=10,
                            show_progress=False)
    assert len(results) == 2
    for src, out, err in results:
        assert err is None, err
        assert out and os.path.isfile(out)
        assert os.path.basename(out).startswith("output_")

"""编辑引擎测试：纯张量操作，无需 FFmpeg。"""
import pytest
import torch

from tensor_editor import TensorEditor, EditError


def _editor(t=20, h=32, w=48, value=100):
    frames = torch.full((t, h, w, 3), value, dtype=torch.uint8)
    return TensorEditor(frames)


def test_cut_frames():
    ed = _editor()
    head = ed.cut_frames(0, 10)
    assert head.shape == (10, 32, 48, 3)
    assert ed.num_frames == 20  # 原编辑器不受影响
    with pytest.raises(EditError):
        ed.cut_frames(15, 10)
    with pytest.raises(EditError):
        ed.cut_frames(0, 999)


def test_modify_region_color_and_invert():
    ed = _editor(value=100)
    ed.modify_region(0, 0, 8, 0, 8, "red")
    assert tuple(ed.frames[0, 0, 0].tolist()) == (255, 0, 0)
    assert tuple(ed.frames[0, 10, 10].tolist()) == (100, 100, 100)  # 区域外不变
    ed.modify_region(1, 0, 8, 0, 8, "invert")
    assert tuple(ed.frames[1, 0, 0].tolist()) == (155, 155, 155)
    with pytest.raises(EditError):
        ed.modify_region(999, 0, 8, 0, 8, "red")
    with pytest.raises(EditError):
        ed.modify_region(0, 0, 8, 0, 8, "notacolor")


def test_concatenate():
    a, b = _editor(t=10), _editor(t=5)
    c = a.concatenate(b)
    assert c.num_frames == 15
    bad = TensorEditor(torch.zeros(5, 16, 48, 3, dtype=torch.uint8))
    with pytest.raises(EditError):
        a.concatenate(bad)


def test_apply_color():
    ed = _editor(value=100)
    ed.apply_color(brightness=2.0)
    assert ed.frames.max().item() == 200
    ed2 = _editor(value=128)
    ed2.apply_color(contrast=0.0)  # 对比度 0 -> 全灰 128
    assert (ed2.frames == 128).all()


def test_apply_transition():
    ed = _editor(t=20, value=200)
    ed.apply_transition("fadein", duration=10)
    assert ed.frames[0].max().item() == 0      # 首帧全黑
    assert ed.frames[19].max().item() == 200   # 尾帧不变
    assert 0 < ed.frames[5].max().item() < 200  # 中间渐变
    with pytest.raises(EditError):
        ed.apply_transition("zoom", 10)


def test_resize():
    ed = _editor(t=4, h=32, w=48)
    ed.resize(24, 16)
    assert ed.shape == (4, 16, 24, 3)


def test_chain():
    ed = _editor(t=30)
    out = (ed.cut_frames(0, 20)
             .apply_color(brightness=1.2)
             .apply_transition("fadeout", duration=5))
    assert out.num_frames == 20
    assert out.frames[-1].max().item() == 0  # 淡出尾帧全黑


def test_add_text():
    ed = _editor(t=2, h=64, w=96, value=0)
    ed.add_text((5, 5), "Hi", font_size=16, frame_idx=0)
    assert ed.frames[0].sum().item() > 0   # 第 0 帧有字
    assert ed.frames[1].sum().item() == 0  # 第 1 帧无字

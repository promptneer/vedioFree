"""张量编辑引擎：对 [T, H, W, C] uint8 视频张量做切帧、改像素、拼接、调色等操作。

所有操作都在张量层面完成，不经过任何 GUI 或商业软件。
能原地做的尽量原地做（modify_region / apply_color / apply_transition），
结构性操作（cut / concatenate / resize）返回新张量或新编辑器。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ColorLike = Union[str, Tuple[int, int, int], List[int]]

_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


class EditError(Exception):
    """编辑操作失败时抛出。"""


def _parse_color(operation: ColorLike) -> Tuple[int, int, int] | str:
    """解析 modify_region 的 operation 参数，返回 RGB 元组或 'invert'。"""
    if isinstance(operation, str):
        if operation == "invert":
            return "invert"
        if operation in _NAMED_COLORS:
            return _NAMED_COLORS[operation]
        raise EditError(
            f"未知颜色名: {operation}，可用: invert / {sorted(_NAMED_COLORS)} / RGB元组"
        )
    if isinstance(operation, (tuple, list)) and len(operation) == 3:
        try:
            rgb = tuple(int(v) for v in operation)
        except (TypeError, ValueError) as exc:
            raise EditError(f"RGB 取值非法: {operation}") from exc
        if any(v < 0 or v > 255 for v in rgb):
            raise EditError(f"RGB 取值须在 0-255: {operation}")
        return rgb
    raise EditError(f"operation 非法: {operation!r}")


class TensorEditor:
    """视频张量编辑器，包装 [T, H, W, C] uint8 张量，支持链式调用。

    Example:
        >>> editor = TensorEditor(frames)
        >>> (editor.cut_frames(0, 100)
        ...        .apply_color(brightness=1.1)
        ...        .apply_transition("fade", duration=10))
    """

    def __init__(self, frames: torch.Tensor) -> None:
        if not isinstance(frames, torch.Tensor):
            raise EditError(f"frames 须为 torch.Tensor，实际 {type(frames)}")
        if frames.ndim != 4 or frames.shape[3] not in (1, 3, 4):
            raise EditError(
                f"frames 形状应为 [T,H,W,C] 且 C∈{{1,3,4}}，实际 {tuple(frames.shape)}"
            )
        if frames.dtype != torch.uint8:
            raise EditError(f"frames dtype 应为 uint8，实际 {frames.dtype}")
        if frames.shape[0] == 0:
            raise EditError("frames 为空（T=0）")
        self.frames = frames

    # ---------- 基础属性 ----------

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """返回 (T, H, W, C)。"""
        return tuple(self.frames.shape)  # type: ignore[return-value]

    @property
    def num_frames(self) -> int:
        """总帧数 T。"""
        return self.frames.shape[0]

    # ---------- 切帧 / 拼接 / 重排 ----------

    def cut_frames(self, start: int, end: int) -> "TensorEditor":
        """切出 [start, end) 帧，返回新编辑器（与原张量共享内存的视图）。

        Raises:
            EditError: 下标越界或 start >= end。
        """
        t = self.num_frames
        if not (0 <= start < t):
            raise EditError(f"start 越界: {start}，有效范围 [0, {t})")
        if not (0 < end <= t):
            raise EditError(f"end 越界: {end}，有效范围 (0, {t}]")
        if start >= end:
            raise EditError(f"start({start}) 须小于 end({end})")
        logger.debug("cut_frames: [%d, %d)", start, end)
        return TensorEditor(self.frames[start:end])

    def concatenate(self, other: Union["TensorEditor", torch.Tensor]) -> "TensorEditor":
        """在时间维拼接另一个张量/编辑器，返回新编辑器。

        要求 H、W、C 一致。
        """
        other_frames = other.frames if isinstance(other, TensorEditor) else other
        if not isinstance(other_frames, torch.Tensor):
            raise EditError("concatenate 参数须为 TensorEditor 或 torch.Tensor")
        if other_frames.shape[1:] != self.frames.shape[1:]:
            raise EditError(
                f"拼接形状不一致: {tuple(self.frames.shape)} vs {tuple(other_frames.shape)}"
                "（H、W、C 须相同）"
            )
        if other_frames.dtype != torch.uint8:
            raise EditError("拼接张量 dtype 须为 uint8")
        merged = torch.cat([self.frames, other_frames], dim=0)
        logger.debug("concatenate: %d + %d 帧", self.num_frames, other_frames.shape[0])
        return TensorEditor(merged)

    def permute(self, dims: Sequence[int]) -> "TensorEditor":
        """按 dims 重排维度，返回新编辑器（视图，不拷贝）。

        例如 ``permute((0, 3, 1, 2))`` 得到 [T, C, H, W]。
        注意：重排后不再是标准视频布局，后续视频操作前请排回来。
        """
        if sorted(dims) != [0, 1, 2, 3]:
            raise EditError(f"dims 须为 0..3 的排列，实际 {list(dims)}")
        return TensorEditor(self.frames.permute(*dims))

    # ---------- 像素修改 ----------

    def modify_region(
        self,
        frame_idx: int,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        operation: ColorLike = "black",
    ) -> "TensorEditor":
        """原地修改某一帧的矩形区域。

        Args:
            frame_idx: 帧下标。
            y1, y2: 行范围 [y1, y2)。
            x1, x2: 列范围 [x1, x2)。
            operation: ``"invert"``（反色）、颜色名
                （black/white/red/green/blue/yellow）或 RGB 三元组。

        Returns:
            self（支持链式调用）。
        """
        t, h, w, _ = self.frames.shape
        if not (0 <= frame_idx < t):
            raise EditError(f"frame_idx 越界: {frame_idx}，有效范围 [0, {t})")
        if not (0 <= y1 < y2 <= h):
            raise EditError(f"行范围非法: [{y1}, {y2})，H={h}")
        if not (0 <= x1 < x2 <= w):
            raise EditError(f"列范围非法: [{x1}, {x2})，W={w}")
        op = _parse_color(operation)
        region = self.frames[frame_idx, y1:y2, x1:x2]
        if op == "invert":
            region.copy_(255 - region)
        else:
            region.copy_(torch.tensor(op, dtype=torch.uint8, device=region.device))
        logger.debug("modify_region: frame=%d box=(%d,%d,%d,%d) op=%s",
                     frame_idx, y1, y2, x1, x2, operation)
        return self

    # ---------- 调色 ----------

    def apply_color(
        self,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> "TensorEditor":
        """调色（原地写回，保持同一张量对象）。

        Args:
            brightness: 亮度倍数，1.0 为不变。
            contrast: 对比度倍数，1.0 为不变。
            saturation: 饱和度倍数，1.0 为不变（仅 RGB 生效）。
        """
        if self.frames.shape[3] != 3:
            raise EditError("apply_color 仅支持 RGB（C=3）张量")
        x = self.frames.float() / 255.0
        if brightness != 1.0:
            x = x * brightness
        if contrast != 1.0:
            x = (x - 0.5) * contrast + 0.5
        if saturation != 1.0:
            luma = x @ torch.tensor([0.299, 0.587, 0.114],
                                    dtype=x.dtype, device=x.device)
            luma = luma.unsqueeze(-1)
            x = luma + saturation * (x - luma)
        x = x.clamp(0.0, 1.0)
        self.frames.copy_((x * 255.0).round().to(torch.uint8))
        logger.debug("apply_color: b=%s c=%s s=%s", brightness, contrast, saturation)
        return self

    # ---------- 转场 ----------

    def apply_transition(self, type: str = "fade", duration: int = 10) -> "TensorEditor":
        """转场（原地写回）。

        Args:
            type: ``"fadein"``（从黑场淡入）、``"fadeout"``（淡出到黑场）、
                ``"fade"``（淡入+淡出）。
            duration: 转场帧数（会被钳制到 [1, T]）。
        """
        if type not in ("fadein", "fadeout", "fade"):
            raise EditError(f"未知转场类型: {type}，可用 fadein/fadeout/fade")
        t = self.num_frames
        d = max(1, min(int(duration), t))
        x = self.frames.float()
        if type in ("fadein", "fade"):
            ramp = torch.linspace(0, 1, d, device=x.device).view(d, 1, 1, 1)
            x[:d] = x[:d] * ramp
        if type in ("fadeout", "fade"):
            ramp = torch.linspace(1, 0, d, device=x.device).view(d, 1, 1, 1)
            x[-d:] = x[-d:] * ramp
        self.frames.copy_(x.round().clamp(0, 255).to(torch.uint8))
        logger.debug("apply_transition: type=%s duration=%d", type, d)
        return self

    # ---------- 字幕 ----------

    def add_text(
        self,
        position: Tuple[int, int],
        text: str,
        font_size: int = 24,
        color: Tuple[int, int, int] = (255, 255, 255),
        frame_idx: Optional[int] = None,
    ) -> "TensorEditor":
        """加字幕（原地写回）。

        Args:
            position: 左上角 (x, y) 像素坐标。
            text: 文本内容。
            font_size: 字号。
            color: RGB 颜色。
            frame_idx: 只画某一帧；None 表示所有帧。

        需要 pillow，未安装会抛 EditError 提示安装。
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise EditError("add_text 需要 pillow，请 pip install pillow") from exc

        if self.frames.shape[3] != 3:
            raise EditError("add_text 仅支持 RGB（C=3）张量")
        t = self.num_frames
        indices = range(t) if frame_idx is None else [frame_idx]
        for i in indices:
            if not (0 <= i < t):
                raise EditError(f"frame_idx 越界: {i}")
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # 旧版 pillow 不支持 size 参数
            font = ImageFont.load_default()
        for i in indices:
            arr = self.frames[i].cpu().numpy()
            img = Image.fromarray(np.ascontiguousarray(arr), mode="RGB")
            ImageDraw.Draw(img).text(position, text, font=font, fill=tuple(color))
            self.frames[i] = torch.as_tensor(np.array(img))
        logger.debug("add_text: %r at %s frames=%s", text, position,
                     "all" if frame_idx is None else frame_idx)
        return self

    # ---------- 缩放 ----------

    def resize(self, width: int, height: int) -> "TensorEditor":
        """缩放到指定分辨率（双线性插值），返回 self（张量被替换）。

        注意：这是少数几个会产生新张量（一次拷贝）的操作之一。
        """
        if width <= 0 or height <= 0:
            raise EditError(f"分辨率非法: {width}x{height}")
        x = self.frames.permute(0, 3, 1, 2).float()  # [T,C,H,W]
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        self.frames = x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        logger.debug("resize: -> %dx%d", width, height)
        return self

    # ---------- 保存 ----------

    def save(self, path: str) -> "TensorEditor":
        """保存中间结果。

        - ``.pt`` / ``.pth``：torch.save 存张量。
        - 其他后缀：按视频编码（mp4），需 fps 参数时用 save_video。
        """
        from .encode import encode_tensor_to_video

        lower = path.lower()
        if lower.endswith((".pt", ".pth")):
            torch.save(self.frames, path)
        else:
            encode_tensor_to_video(self.frames, path)
        logger.info("save: %s", path)
        return self

    def save_video(self, path: str, fps: int = 30) -> "TensorEditor":
        """按指定帧率把当前张量编码为视频。"""
        from .encode import encode_tensor_to_video

        encode_tensor_to_video(self.frames, path, fps=fps)
        return self

    def clone(self) -> "TensorEditor":
        """深拷贝一份编辑器（张量独立）。"""
        return TensorEditor(self.frames.clone())

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:  # pragma: no cover
        return f"TensorEditor(shape={tuple(self.frames.shape)}, dtype={self.frames.dtype})"

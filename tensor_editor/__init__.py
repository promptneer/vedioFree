"""AI 原生视频张量编辑引擎：视频 <-> PyTorch 张量，全程张量层面操作。"""

from .decode import decode_video_to_tensor, DecodeError
from .encode import encode_tensor_to_video, EncodeError
from .edit_engine import TensorEditor, EditError

__all__ = [
    "decode_video_to_tensor",
    "DecodeError",
    "encode_tensor_to_video",
    "EncodeError",
    "TensorEditor",
    "EditError",
]

__version__ = "0.1.0"

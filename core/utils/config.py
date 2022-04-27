from core.utils.types import PathLike
from dataclasses import dataclass


@dataclass
class LipNetConfig:
    model_weight_path: PathLike
    video_path: PathLike
    frame_count: int = (75,)
    image_channels: int = (3,)
    image_height: int = (50,)
    image_width: int = (100,)
    max_string: int = (32,)
    output_size: int = (28,)

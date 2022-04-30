from dataclasses import dataclass


@dataclass(repr=True)
class LipNetConfig:
    frame_count: int = 75
    image_channels: int = 3
    image_height: int = 50
    image_width: int = 100
    max_string: int = 32
    output_size: int = 28


@dataclass(repr=True)
class DenseNetConfig:
    batch_size: int = 75
    image_channels: int = 3
    image_height: int = 244
    image_width: int = 244

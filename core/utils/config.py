from dataclasses import dataclass


@dataclass(repr=True)
class LipNetConfig:
    frame_count: int = 75
    image_height: int = 50
    image_width: int = 100
    image_channels: int = 3
    max_string: int = 32
    output_size: int = 28


@dataclass(repr=True)
class BaselineConfig:
    batch_size: int = 75
    image_channels: int = 3
    image_height: int = 224
    image_width: int = 224


emotion = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

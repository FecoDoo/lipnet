import numpy as np
from matplotlib import patheffects as path_effects, pyplot as plt
from core.utils.types import Stream

FRAME_RATE = 1 / 25


def visualize_video_subtitle(stream: Stream, subtitle: str, swap_axes: bool = False):
    if swap_axes:
        stream = np.swapaxes(stream, 1, 2)

    ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.axis("off")

    text = plt.text(
        0.5,
        0.1,
        "",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontdict={"fontsize": 14, "color": "yellow", "fontweight": 500},
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="black"), path_effects.Normal()]
    )

    subs = subtitle.split()
    inc = max(len(stream) / (len(subs) + 1), 0.01)

    img = None

    for i, frame in enumerate(stream):
        sub = " ".join(subs[: int(i / inc)])
        text.set_text(sub)

        if img is None:
            img = plt.imshow(frame)
        else:
            img.set_data(frame)

        plt.pause(FRAME_RATE)

    plt.show()

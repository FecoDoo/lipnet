# Helper functions to transform between text<->labels
# Source: https://github.com/rizkiarm/LipNet/blob/master/lipnet/lipreading/helpers.py
import numpy as np
from core.utils.types import Labels


def text_to_labels(text: str) -> Labels:
    ret = []
    for char in text:
        if "a" <= char <= "z":
            ret.append(ord(char) - ord("a"))
        elif char == " ":
            ret.append(26)

    return np.array(ret, dtype=np.int16)


def labels_to_text(labels: Labels) -> str:
    # 26 is space, 27 is CTC blank char
    text = ""
    for c in labels:
        if 0 <= c < 26:
            text += chr(c + ord("a"))
        elif c == 26:
            text += " "

    return text

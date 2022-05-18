import os
from dataclasses import dataclass
import numpy as np
from core.utils.label import text_to_labels
from core.utils.types import Labels, Path

__SILENCE_TOKENS = ["sp", "sil"]


@dataclass()
class Align:
    """Align object"""

    sentence: str
    labels: Labels
    length: int


def align_from_file(align_path: Path, max_string: int) -> Align:
    """Load align info from file

    Args:
        align_path (Path): path to .align file
        max_string (int): the maximum amount of characters to expect as the encoded align sentence vector

    Returns:
        Align: Align object
    """
    with open(align_path, "r") as fp:
        lines = fp.readlines()

    # generate matrix of align
    align = [
        (int(tokens[0]) / 1000, int(tokens[1]) / 1000, tokens[2])
        for tokens in [line.strip().split(" ") for line in lines]
    ]
    align = __strip_from_align(align, __SILENCE_TOKENS)

    sentence = __get_align_sentence(align, __SILENCE_TOKENS)
    labels = __get_sentence_labels(sentence)
    padded_labels = __get_padded_label(labels, max_string)

    return Align(sentence, padded_labels, padded_labels.shape[0])


def __strip_from_align(align: list, items: list) -> list:
    """
    Remove lines which indicates silence periods
    """
    return [sub for sub in align if sub[-1] not in items]


def __get_align_sentence(align: list, items: list) -> str:
    """
    Generate whole sentence from align file
    """
    return " ".join([y[-1] for y in align if y[-1] not in items])


def __get_sentence_labels(sentence: str) -> Labels:
    """Convert sentence into list of labels

    Args:
        sentence (str): sentence

    Returns:
        List[int]: list of converted labels
    """
    return text_to_labels(sentence)


def __get_padded_label(labels: Labels, max_string: int = 12) -> Labels:
    """Pad list of labels with -1 until the length reaches max_string

    Args:
        labels (list): list of labels
        max_string (int): max length of the list

    Returns:
        np.ndarray: _description_
    """
    return np.pad(
        labels, (0, max_string - labels.shape[0]), mode="constant", constant_values=-1
    )

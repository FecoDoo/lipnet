import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from more_itertools import unzip
from typing import Tuple
from core.utils.types import Stream
from common.facemesh import recognition

def detecting(frame):
    """detecting face and lip area from image

    Args:
        frame (np.array): numpy array with shape H x W x C

    Returns:
        face: numpy array of cropped face image
        lip: numpy array of cropped lip image
    """
    res = recognition(frame)

    face = res[0] if res[0] is not None and 0 not in res[0].shape else None
    lip = res[1] if res[0] is not None and 0 not in res[0].shape else None

    return face, lip

def cleaning(stream):
    """Remove null values after face detection process

    Args:
        stream (_type_): _description_

    Returns:
        _type_: _description_
    """
    return pd.Series(data=stream).fillna(method="backfill").to_numpy()


def resize_face(frame, dsize=(224,224)):
    if len(dsize) != 2:
        raise ValueError("target size should be in 2 dimension")

    return cv2.resize(frame, dsize=dsize[::-1])

def resize_lip(frame, dsize=(50,100)):
    if len(dsize) != 2:
        raise ValueError("target size should be in 2 dimension")

    return cv2.resize(frame, dsize=dsize[::-1])



def extracting(stream: Stream) -> Tuple[Stream, Stream]:
    """Detecting faces and lips from video stream

    Args:
        stream (Stream): video stream in numpy array format

    Returns:
        faces: stream containing detected and resized faces
        lips: stream containing detected and resized lips
    """
    faces, lips = None, None

    with Pool(os.cpu_count() - 1) as pool:
        results = pool.map_async(detecting, stream).get()

        faces, lips = unzip(results)
        faces, lips = cleaning(list(faces)), cleaning(list(lips))

        faces = pool.map_async(resize_face, faces).get()
        lips = pool.map_async(resize_lip, lips).get()

    return np.array(faces, dtype=np.uint8), np.array(lips, dtype=np.uint8)
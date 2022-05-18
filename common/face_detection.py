import mediapipe as mp
import numpy as np
import cv2
from core.utils.types import Frame, Tuple, List

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

all_landmarks = range(0, 468)

lip_landmarks = {"left": 138, "right": 367, "bottom": 199, "top": 1}
face_landmarks = {"left": 127, "right": 356, "bottom": 152, "top": 10}


def get_relative_coordinates(landmark):
    return np.array([landmark.y, landmark.x], dtype=np.float32)


def get_absolute_coordinates(shape, landmark):
    relative_coor = get_relative_coordinates(landmark)

    absolute_coor = (shape * relative_coor).astype(np.int16)

    return absolute_coor


def crop(frame: Frame, box: List[int]) -> Frame:
    """crop frame according to given box

    Args:
        frame (Frame): Frame, H x W x C
        box (Tuple[int]): box

    Returns:
        Frame: Frame, H x W x C
    """

    return frame[box[0] : box[2], box[1] : box[3]]


def get_box(frame, all_landmarks, landmarks) -> Frame:
    """crop image according to given landmarks

    Args:
        frame (Frame): image
        all_landmarks (_type_): landmarks predicted by mediapipe
        landmarks (_type_): landmarks at the middle of bounding box's edges

    Returns:
        Frame: cropped image
    """
    left = get_absolute_coordinates(frame.shape[:2], all_landmarks[landmarks["left"]])
    right = get_absolute_coordinates(frame.shape[:2], all_landmarks[landmarks["right"]])
    top = get_absolute_coordinates(frame.shape[:2], all_landmarks[landmarks["top"]])
    bottom = get_absolute_coordinates(
        frame.shape[:2], all_landmarks[landmarks["bottom"]]
    )

    top_left = np.array([top[0], left[1]], dtype=np.int16)
    bottom_right = np.array([bottom[0], right[1]], dtype=np.int16)

    box = np.append(arr=top_left, values=bottom_right)

    return crop(frame, box)


def draw_points(frame, landmarks, points):
    """draw all points with annotatation

    Args:
        frame (_type_): _description_
        landmarks (_type_): _description_
        points (_type_): _description_

    Returns:
        _type_: _description_
    """
    annotated_frame = frame.copy()
    shape = frame.shape[:2]
    for idx in points:
        point_coor = get_absolute_coordinates(shape, landmarks[idx])
        annotated_frame = cv2.putText(
            annotated_frame,
            str(idx),
            point_coor[::-1],
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
            thickness=1,
            color=(250, 225, 100),
        )
    return annotated_frame


def segmentation(frame, detected_landmarks):
    """process raw frame, return cropped face image and lip image

    Args:
        frame (Frame): raw Frame
        detected_landmarks (_type_): landmarks predicted by mediapipe
    """

    lip = get_box(frame, detected_landmarks, lip_landmarks)
    face = get_box(frame, detected_landmarks, face_landmarks)
    return face, lip


def recognition(frame: Frame) -> Tuple[Frame]:
    """crop face and lip area from the given frame

    Args:
        frame (Frame): image of video sequence

    Returns:
        Tuple[Frame]: face & lip area
    """
    face = None
    lip = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(frame)

        if not results.multi_face_landmarks:
            print("no face detected")
            return None, None

        face_crop, lip_crop = segmentation(
            frame, results.multi_face_landmarks[0].landmark
        )

    return face_crop, lip_crop

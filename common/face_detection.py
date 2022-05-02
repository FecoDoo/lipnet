import mediapipe as mp

from core.utils.types import Frame, Tuple

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def get_face_box(n_row: int, n_col: int, location) -> Tuple[int]:
    """generate bounding box positions for detected face

    Args:
        n_row (int): frame height
        n_col (int): frame width
        location (_type_): mediapipe location object

    Returns:
        Tuple: box
    """

    relative_bounding_box = location.relative_bounding_box

    rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, n_col, n_row
    )

    rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height,
        n_col,
        n_row,
    )

    return rect_start_point + rect_end_point


def get_lip_box(n_row: int, n_col: int, mouth_keypoint, nose_keypoint) -> Tuple[int]:
    """generate bounding box positions for detected lip

    Args:
        n_row (int): frame height
        n_col (int): frame width
        mouth_keypoint: relative position of mouth center point
        nose_keypoint: relative position of nose tip point

    Returns:
        Tuple: box
    """

    # TO-DO
    # rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
    #     mouth_keypoint.xmin,
    #     mouth_keypoint.ymin,
    #     n_col,
    #     n_row
    # )

    # rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
    #     relative_bounding_box.xmin + relative_bounding_box.width,
    #     relative_bounding_box.ymin + relative_bounding_box.height,
    #     n_col,
    #     n_row
    # )

    # return rect_start_point + rect_end_point

    return None


def process(frame: Frame, detection):
    """process raw frame, return cropped face image and lip image

    Args:
        frame (Frame): raw Frame
        detection (_type_): mediapipe face detection object
    """
    location = detection.location_data

    # get face box
    face_box = get_face_box(
        n_row=frame.shape[0], n_col=frame.shape[1], location=location
    )

    # to-do: return lip cropping
    # get mouth keypoint
    # get relative position of mouth center

    mouth_keypoint = mp_face_detection.get_key_point(
        detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER
    )

    nose_keypoint = mp_face_detection.get_key_point(
        detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
    )

    lip_box = get_lip_box(
        n_row=frame.shape[0],
        n_col=frame.shape[1],
        mouth_keypoint=mouth_keypoint,
        nose_keypoint=nose_keypoint,
    )
    # keypoint_px = mp_drawing._normalized_to_pixel_coordinates(keypoint.x, keypoint.y, frame.shape[0], frame.shape[1])

    return face_box


def crop(frame: Frame, box: Tuple[int]) -> Frame:
    """crop frame according to given box

    Args:
        frame (Frame): Frame, H x W x C
        box (Tuple[int]): box

    Returns:
        Frame: Frame, H x W x C
    """

    return frame[box[1] : box[3], box[0] : box[2]]


def recognition(frame: Frame) -> Frame:
    face = None
    lip = None

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.8
    ) as face_detection:
        results = face_detection.process(frame)

        if results.detections is None:
            return None

        for detection in results.detections:
            box = process(frame, detection)

            face = crop(frame=frame, box=box)

    return face

import argparse
import math
import time
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# Landmark indexes
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


# MCP, PIP, DIP, TIP for the four non-thumb fingers
FINGER_LANDMARKS = [
    (5, 6, 7, 8),       # Index
    (9, 10, 11, 12),    # Middle
    (13, 14, 15, 16),   # Ring
    (17, 18, 19, 20),   # Pinky
]


# Recent counts are stored to reduce flickering
finger_history = {
    "Left": deque(maxlen=5),
    "Right": deque(maxlen=5),
    "Unknown": deque(maxlen=5),
}


def distance_3d(point_a, point_b):
    """Return the 3D distance between two landmarks."""
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate angle ABC in degrees.

    point_b is the middle point of the angle.
    """
    vector_ba = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )

    vector_bc = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )

    dot_product = sum(
        a * b for a, b in zip(vector_ba, vector_bc)
    )

    magnitude_ba = math.sqrt(
        sum(value ** 2 for value in vector_ba)
    )

    magnitude_bc = math.sqrt(
        sum(value ** 2 for value in vector_bc)
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine = dot_product / (magnitude_ba * magnitude_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def is_thumb_extended(hand_landmarks):
    """
    Return True only when the thumb is genuinely extended outward.

    This avoids counting a thumb folded across the palm.
    """
    wrist = hand_landmarks[WRIST]

    thumb_cmc = hand_landmarks[THUMB_CMC]
    thumb_mcp = hand_landmarks[THUMB_MCP]
    thumb_ip = hand_landmarks[THUMB_IP]
    thumb_tip = hand_landmarks[THUMB_TIP]

    index_mcp = hand_landmarks[INDEX_MCP]
    middle_mcp = hand_landmarks[MIDDLE_MCP]
    pinky_mcp = hand_landmarks[PINKY_MCP]

    # Palm width and length normalize the thresholds
    palm_width = distance_3d(index_mcp, pinky_mcp)
    palm_length = distance_3d(wrist, middle_mcp)

    if palm_width <= 0 or palm_length <= 0:
        return False

    # Check whether the thumb joints are reasonably straight
    thumb_mcp_angle = calculate_angle(
        thumb_cmc,
        thumb_mcp,
        thumb_ip,
    )

    thumb_ip_angle = calculate_angle(
        thumb_mcp,
        thumb_ip,
        thumb_tip,
    )

    # Check whether the thumb tip is separated from the palm
    tip_to_index_mcp = distance_3d(
        thumb_tip,
        index_mcp,
    )

    tip_to_middle_mcp = distance_3d(
        thumb_tip,
        middle_mcp,
    )

    tip_to_wrist = distance_3d(
        thumb_tip,
        wrist,
    )

    ip_to_wrist = distance_3d(
        thumb_ip,
        wrist,
    )

    thumb_is_straight = (
        thumb_mcp_angle > 125
        and thumb_ip_angle > 150
    )

    thumb_is_away_from_index = (
        tip_to_index_mcp > palm_width * 0.85
    )

    thumb_is_away_from_palm_center = (
        tip_to_middle_mcp > palm_width * 0.80
    )

    thumb_tip_extends_past_ip = (
        tip_to_wrist > ip_to_wrist + palm_length * 0.03
    )

    return (
        thumb_is_straight
        and thumb_is_away_from_index
        and thumb_is_away_from_palm_center
        and thumb_tip_extends_past_ip
    )


def is_finger_extended(
    hand_landmarks,
    mcp_index,
    pip_index,
    dip_index,
    tip_index,
):
    """
    Check whether a non-thumb finger is extended.

    Joint angles make this more reliable than checking only the y-position.
    """
    wrist = hand_landmarks[WRIST]
    middle_mcp = hand_landmarks[MIDDLE_MCP]

    mcp = hand_landmarks[mcp_index]
    pip = hand_landmarks[pip_index]
    dip = hand_landmarks[dip_index]
    tip = hand_landmarks[tip_index]

    palm_length = distance_3d(wrist, middle_mcp)

    if palm_length <= 0:
        return False

    pip_angle = calculate_angle(mcp, pip, dip)
    dip_angle = calculate_angle(pip, dip, tip)

    tip_to_wrist = distance_3d(tip, wrist)
    pip_to_wrist = distance_3d(pip, wrist)

    finger_is_straight = (
        pip_angle > 150
        and dip_angle > 145
    )

    fingertip_is_extended = (
        tip_to_wrist
        > pip_to_wrist + palm_length * 0.08
    )

    return finger_is_straight and fingertip_is_extended


def count_fingers(hand_landmarks):
    """Return the number of raised fingers from 0 to 5."""
    fingers_up = 0

    # Thumb
    if is_thumb_extended(hand_landmarks):
        fingers_up += 1

    # Index, middle, ring, and pinky
    for mcp, pip, dip, tip in FINGER_LANDMARKS:
        if is_finger_extended(
            hand_landmarks,
            mcp,
            pip,
            dip,
            tip,
        ):
            fingers_up += 1

    return fingers_up


def smooth_count(hand_label, current_count):
    """Reduce rapid changes between consecutive frames."""
    if hand_label not in finger_history:
        finger_history[hand_label] = deque(maxlen=5)

    finger_history[hand_label].append(current_count)

    history = list(finger_history[hand_label])

    # Choose the most frequent count in recent frames
    return max(
        set(history),
        key=lambda value: (
            history.count(value),
            history[::-1].index(value) * -1,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time finger counting using MediaPipe and OpenCV."
    )

    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index, video path, or IP camera URL. Default: 0",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera frame width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera frame height.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to hand_landmarker.task.",
    )

    return parser.parse_args()


def open_camera(camera_source, width, height):
    source = (
        int(camera_source)
        if camera_source.isdigit()
        else camera_source
    )

    # CAP_DSHOW usually opens Windows webcams faster
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(source)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture


def ensure_model(model_path):
    model_path = Path(model_path)

    if model_path.exists():
        return model_path

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading MediaPipe model to {model_path}...")

    try:
        urlretrieve(
            MODEL_URL,
            model_path,
        )

    except Exception as error:
        raise RuntimeError(
            "Could not download hand_landmarker.task. "
            f"Download it manually from {MODEL_URL} "
            f"and save it as {model_path}."
        ) from error

    print("Model downloaded successfully.")

    return model_path


def draw_hand_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]

    for start_index, end_index in HAND_CONNECTIONS:
        start_point = hand_landmarks[start_index]
        end_point = hand_landmarks[end_index]

        cv2.line(
            frame,
            (
                int(start_point.x * width),
                int(start_point.y * height),
            ),
            (
                int(end_point.x * width),
                int(end_point.y * height),
            ),
            (0, 255, 0),
            2,
        )

    for landmark in hand_landmarks:
        cv2.circle(
            frame,
            (
                int(landmark.x * width),
                int(landmark.y * height),
            ),
            4,
            (0, 0, 255),
            -1,
        )


def main():
    args = parse_args()

    model_path = ensure_model(args.model)

    capture = open_camera(
        args.camera,
        args.width,
        args.height,
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check the webcam, "
            "camera index, or IP camera URL."
        )

    previous_time = 0.0
    start_time = time.monotonic()

    base_options = python.BaseOptions(
        model_asset_path=str(model_path)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = capture.read()

                if not success:
                    print("Could not read a frame from the camera.")
                    break

                # Mirror the image
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.hand_landmarks:
                    for index, hand_landmarks in enumerate(
                        result.hand_landmarks
                    ):
                        draw_hand_landmarks(
                            frame,
                            hand_landmarks,
                        )

                        hand_label = "Unknown"

                        if index < len(result.handedness):
                            category = result.handedness[index][0]

                            hand_label = (
                                category.category_name
                                or category.display_name
                                or "Unknown"
                            )

                        raw_count = count_fingers(
                            hand_landmarks
                        )

                        finger_count = smooth_count(
                            hand_label,
                            raw_count,
                        )

                        text_y = 70 + index * 65

                        cv2.putText(
                            frame,
                            f"{hand_label}: {finger_count}",
                            (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3,
                        )

                current_time = time.monotonic()

                if previous_time:
                    elapsed = current_time - previous_time
                    fps = 1.0 / elapsed if elapsed > 0 else 0
                else:
                    fps = 0

                previous_time = current_time

                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow(
                    "Finger Count (0-5)",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np


def calculate_motion_score(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    previous_gray = None
    scores = []
    current_frame = start_frame

    while current_frame < end_frame:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_gray is not None:
            difference = cv2.absdiff(previous_gray, gray)
            score = np.mean(difference)
            scores.append(score)

        previous_gray = gray
        current_frame += 1

    cap.release()

    if len(scores) == 0:
        return 0

    return float(np.mean(scores))
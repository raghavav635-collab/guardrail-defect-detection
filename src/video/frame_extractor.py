from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractedFrame:
    frame_number: int
    timestamp_seconds: float
    frame: np.ndarray


def iterate_sampled_frames(
    video_path: Path,
    fps: float,
    sample_seconds: float,
) -> Iterator[ExtractedFrame]:
    """
    Yield one frame at the configured time interval.
    """

    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be greater than zero.")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_interval = max(1, int(round(fps * sample_seconds)))
    frame_number = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_number % frame_interval == 0:
                timestamp_seconds = frame_number / fps

                yield ExtractedFrame(
                    frame_number=frame_number,
                    timestamp_seconds=round(timestamp_seconds, 3),
                    frame=frame,
                )

            frame_number += 1
    finally:
        capture.release()
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    path: str
    filename: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_seconds: float
    file_size_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_video(video_path: Path) -> VideoMetadata:
    """
    Validate a video and return its basic technical metadata.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    supported_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    if video_path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported video type: {video_path.suffix}. "
            f"Supported formats: {sorted(supported_extensions)}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise RuntimeError("The video FPS could not be determined.")

    if total_frames <= 0:
        raise RuntimeError("The video does not contain readable frames.")

    duration_seconds = total_frames / fps
    file_size_mb = video_path.stat().st_size / (1024 * 1024)

    return VideoMetadata(
        path=str(video_path.resolve()),
        filename=video_path.name,
        fps=round(fps, 3),
        total_frames=total_frames,
        width=width,
        height=height,
        duration_seconds=round(duration_seconds, 3),
        file_size_mb=round(file_size_mb, 3),
    )
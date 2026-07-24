import argparse
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "data" / "extracted_frames"


def calculate_blur_score(frame) -> float:
    """
    Measure image sharpness.

    A low score usually means the frame is blurry.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return float(score)


def extract_frames(
    video_path: Path,
    output_folder: Path,
    sample_seconds: float,
    blur_threshold: float,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    accepted_folder = output_folder / "accepted"
    blurry_folder = output_folder / "rejected_blurry"

    accepted_folder.mkdir(parents=True, exist_ok=True)
    blurry_folder.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        video_capture.release()
        raise RuntimeError("Could not determine video FPS.")

    frame_interval = max(1, int(fps * sample_seconds))

    frame_number = 0
    accepted_count = 0
    rejected_count = 0
    metadata = []

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            timestamp_seconds = frame_number / fps
            blur_score = calculate_blur_score(frame)

            filename = (
                f"{video_path.stem}_"
                f"frame_{frame_number:08d}_"
                f"time_{timestamp_seconds:010.2f}.jpg"
            )

            is_accepted = blur_score >= blur_threshold

            if is_accepted:
                output_path = accepted_folder / filename
                accepted_count += 1
            else:
                output_path = blurry_folder / filename
                rejected_count += 1

            saved = cv2.imwrite(str(output_path), frame)

            if not saved:
                video_capture.release()
                raise RuntimeError(f"Could not save frame: {output_path}")

            metadata.append(
                {
                    "source_video": video_path.name,
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp_seconds, 2),
                    "blur_score": round(blur_score, 2),
                    "accepted": is_accepted,
                    "output_path": str(output_path),
                }
            )

        frame_number += 1

    video_capture.release()

    metadata_file = output_folder / "frame_metadata.json"

    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nVideo processing completed")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total video frames: {total_frames}")
    print(f"Sampling interval: every {sample_seconds} second(s)")
    print(f"Accepted frames: {accepted_count}")
    print(f"Rejected blurry frames: {rejected_count}")
    print(f"Results saved in: {output_folder}")
    print(f"Metadata saved in: {metadata_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clear frames from a road inspection video."
    )

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help="Folder where extracted frames will be saved.",
    )

    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Extract one frame every specified number of seconds.",
    )

    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Frames below this sharpness score are rejected.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    extract_frames(
        video_path=arguments.video,
        output_folder=arguments.output,
        sample_seconds=arguments.sample_seconds,
        blur_threshold=arguments.blur_threshold,
    )


if __name__ == "__main__":
    main()
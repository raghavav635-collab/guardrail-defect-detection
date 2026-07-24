import json
from collections import Counter
from pathlib import Path

import cv2

from src.video.duplicate_filter import DuplicateFrameFilter
from src.video.frame_extractor import iterate_sampled_frames
from src.video.quality import evaluate_frame_quality
from src.video.validator import validate_video


def process_video(
    video_path: Path,
    output_folder: Path,
    sample_seconds: float = 1.0,
    blur_threshold: float = 100.0,
    minimum_brightness: float = 30.0,
    maximum_brightness: float = 230.0,
    duplicate_threshold: float = 4.0,
) -> dict:
    """
    Run the complete road-inspection video preprocessing pipeline.
    """

    metadata = validate_video(video_path)

    accepted_folder = output_folder / "accepted"
    rejected_blurry_folder = output_folder / "rejected" / "blurry"
    rejected_dark_folder = output_folder / "rejected" / "too_dark"
    rejected_bright_folder = output_folder / "rejected" / "too_bright"
    duplicate_folder = output_folder / "rejected" / "duplicate"

    for folder in [
        accepted_folder,
        rejected_blurry_folder,
        rejected_dark_folder,
        rejected_bright_folder,
        duplicate_folder,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    duplicate_filter = DuplicateFrameFilter(
        difference_threshold=duplicate_threshold
    )

    frame_records = []
    counts = Counter()

    for extracted in iterate_sampled_frames(
        video_path=video_path,
        fps=metadata.fps,
        sample_seconds=sample_seconds,
    ):
        quality = evaluate_frame_quality(
            frame=extracted.frame,
            blur_threshold=blur_threshold,
            minimum_brightness=minimum_brightness,
            maximum_brightness=maximum_brightness,
        )

        is_duplicate, duplicate_difference = (
            duplicate_filter.check_duplicate(extracted.frame)
        )

        filename = (
            f"{video_path.stem}_"
            f"frame_{extracted.frame_number:08d}_"
            f"time_{extracted.timestamp_seconds:010.2f}.jpg"
        )

        if is_duplicate:
            status = "rejected_duplicate"
            destination = duplicate_folder / filename
            counts["rejected_duplicate"] += 1

        elif quality.is_blurry:
            status = "rejected_blurry"
            destination = rejected_blurry_folder / filename
            counts["rejected_blurry"] += 1

        elif quality.is_too_dark:
            status = "rejected_too_dark"
            destination = rejected_dark_folder / filename
            counts["rejected_too_dark"] += 1

        elif quality.is_too_bright:
            status = "rejected_too_bright"
            destination = rejected_bright_folder / filename
            counts["rejected_too_bright"] += 1

        else:
            status = "accepted"
            destination = accepted_folder / filename
            counts["accepted"] += 1

        saved = cv2.imwrite(str(destination), extracted.frame)

        if not saved:
            raise RuntimeError(f"Could not save frame: {destination}")

        frame_records.append(
            {
                "source_video": video_path.name,
                "frame_number": extracted.frame_number,
                "timestamp_seconds": extracted.timestamp_seconds,
                "status": status,
                "saved_path": str(destination),
                "duplicate_difference": duplicate_difference,
                **quality.to_dict(),
            }
        )

    report = {
        "video_metadata": metadata.to_dict(),
        "configuration": {
            "sample_seconds": sample_seconds,
            "blur_threshold": blur_threshold,
            "minimum_brightness": minimum_brightness,
            "maximum_brightness": maximum_brightness,
            "duplicate_threshold": duplicate_threshold,
        },
        "summary": {
            "sampled_frames": len(frame_records),
            **dict(counts),
        },
        "frames": frame_records,
    }

    report_path = output_folder / "processing_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report
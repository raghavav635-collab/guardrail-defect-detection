import argparse
from pathlib import Path

from src.video.pipeline import process_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and preprocess road-inspection video."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_video",
        help="Output directory.",
    )

    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Extract one frame every N seconds.",
    )

    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum sharpness score.",
    )

    parser.add_argument(
        "--minimum-brightness",
        type=float,
        default=30.0,
        help="Minimum acceptable brightness.",
    )

    parser.add_argument(
        "--maximum-brightness",
        type=float,
        default=230.0,
        help="Maximum acceptable brightness.",
    )

    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=4.0,
        help=(
            "Frames with a difference below this value "
            "are considered duplicates."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    report = process_video(
        video_path=args.video,
        output_folder=args.output,
        sample_seconds=args.sample_seconds,
        blur_threshold=args.blur_threshold,
        minimum_brightness=args.minimum_brightness,
        maximum_brightness=args.maximum_brightness,
        duplicate_threshold=args.duplicate_threshold,
    )

    metadata = report["video_metadata"]
    summary = report["summary"]

    print("\nVideo validation")
    print(f"File: {metadata['filename']}")
    print(f"Resolution: {metadata['width']}x{metadata['height']}")
    print(f"FPS: {metadata['fps']}")
    print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
    print(f"File size: {metadata['file_size_mb']:.2f} MB")

    print("\nProcessing summary")
    print(f"Sampled frames: {summary.get('sampled_frames', 0)}")
    print(f"Accepted: {summary.get('accepted', 0)}")
    print(
        "Rejected blurry: "
        f"{summary.get('rejected_blurry', 0)}"
    )
    print(
        "Rejected dark: "
        f"{summary.get('rejected_too_dark', 0)}"
    )
    print(
        "Rejected bright: "
        f"{summary.get('rejected_too_bright', 0)}"
    )
    print(
        "Rejected duplicates: "
        f"{summary.get('rejected_duplicate', 0)}"
    )

    print(f"\nResults saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
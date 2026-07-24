from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_FOLDER = PROJECT_ROOT / "data" / "images"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"


def main() -> None:
    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    images = list(IMAGE_FOLDER.glob("*.jpg"))
    images += list(IMAGE_FOLDER.glob("*.jpeg"))
    images += list(IMAGE_FOLDER.glob("*.png"))

    if not images:
        raise FileNotFoundError(
            f"No images found in {IMAGE_FOLDER}. Add one JPG or PNG road image."
        )

    model = YOLO("yolo11n.pt")

    model.predict(
        source=str(IMAGE_FOLDER),
        conf=0.25,
        save=True,
        project=str(OUTPUT_FOLDER),
        name="first_test",
        exist_ok=True,
    )

    print("Inference completed.")
    print(f"Check results here: {OUTPUT_FOLDER / 'first_test'}")


if __name__ == "__main__":
    main()
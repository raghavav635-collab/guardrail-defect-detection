from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    blur_score: float
    brightness: float
    contrast: float
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool
    accepted: bool
    rejection_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_blur_score(frame: np.ndarray) -> float:
    """
    Calculate sharpness using variance of the Laplacian.

    Lower scores generally indicate blurrier frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(frame: np.ndarray) -> float:
    """
    Return average grayscale brightness from 0 to 255.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def calculate_contrast(frame: np.ndarray) -> float:
    """
    Return grayscale standard deviation as a simple contrast metric.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def evaluate_frame_quality(
    frame: np.ndarray,
    blur_threshold: float = 100.0,
    minimum_brightness: float = 30.0,
    maximum_brightness: float = 230.0,
) -> FrameQuality:
    blur_score = calculate_blur_score(frame)
    brightness = calculate_brightness(frame)
    contrast = calculate_contrast(frame)

    is_blurry = blur_score < blur_threshold
    is_too_dark = brightness < minimum_brightness
    is_too_bright = brightness > maximum_brightness

    reasons = []

    if is_blurry:
        reasons.append("blurry")

    if is_too_dark:
        reasons.append("too_dark")

    if is_too_bright:
        reasons.append("too_bright")

    accepted = not reasons
    rejection_reason = ",".join(reasons) if reasons else None

    return FrameQuality(
        blur_score=round(blur_score, 3),
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        accepted=accepted,
        rejection_reason=rejection_reason,
    )
import cv2
import numpy as np


class DuplicateFrameFilter:
    """
    Detect near-duplicate sequential frames using a small grayscale image.

    A lower difference score means two frames are more similar.
    """

    def __init__(
        self,
        difference_threshold: float = 4.0,
        comparison_width: int = 64,
        comparison_height: int = 64,
    ) -> None:
        self.difference_threshold = difference_threshold
        self.comparison_size = (comparison_width, comparison_height)
        self.previous_signature: np.ndarray | None = None

    def _create_signature(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.comparison_size)
        return resized.astype(np.float32)

    def check_duplicate(self, frame: np.ndarray) -> tuple[bool, float | None]:
        current_signature = self._create_signature(frame)

        if self.previous_signature is None:
            self.previous_signature = current_signature
            return False, None

        difference = float(
            np.mean(
                np.abs(
                    current_signature - self.previous_signature
                )
            )
        )

        is_duplicate = difference < self.difference_threshold

        if not is_duplicate:
            self.previous_signature = current_signature

        return is_duplicate, round(difference, 3)
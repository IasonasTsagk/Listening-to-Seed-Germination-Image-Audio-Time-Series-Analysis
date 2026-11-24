"""Image loading, validation, colour conversion, and frame-difference helpers."""
from pathlib import Path
import cv2
import numpy as np


def validate_image(image: np.ndarray, expected_size: tuple[int, int] | None = None) -> None:
    """Validate an 8-bit grayscale or BGR image without changing it.

    ``expected_size`` is ``(width, height)`` when a fixed camera layout has
    been independently established. Values outside the 0--255 range or a
    non-three-channel colour image are rejected early.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a non-empty NumPy array.")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"Expected grayscale or three-channel image, got shape {image.shape}.")
    if image.dtype != np.uint8 or image.min() < 0 or image.max() > 255:
        raise ValueError("Expected uint8 pixel values in the range 0--255.")
    if expected_size and (image.shape[1], image.shape[0]) != expected_size:
        raise ValueError(f"Expected {expected_size} (width, height), got {(image.shape[1], image.shape[0])}.")


def load_image(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Load and validate one raw image; never writes to the source path."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        validate_image(image, expected_size)
    return image


def color_spaces(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return RGB, grayscale, HSV and LAB analysis copies from a BGR OpenCV image."""
    validate_image(image)
    return {
        "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "hsv": cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
    }


def split_channels(image: np.ndarray, space: str) -> dict[str, np.ndarray]:
    """Return named HSV or LAB channels for exploratory visualization."""
    labels = {"hsv": ("H", "S", "V"), "lab": ("L", "a", "b")}
    if space not in labels:
        raise ValueError("space must be 'hsv' or 'lab'.")
    return dict(zip(labels[space], cv2.split(image), strict=True))


def absolute_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute an unsigned absolute pixel difference for identically sized images."""
    if first.shape != second.shape:
        raise ValueError(f"Image shape mismatch: {first.shape} vs {second.shape}")
    return cv2.absdiff(first, second)


def difference_features(difference: np.ndarray, thresholds: tuple[float, ...]) -> dict:
    """Summarise one difference image; percentages are per-pixel, not germination labels."""
    values = difference.astype(np.float32)
    features = {"mean_difference": float(values.mean()), "median_difference": float(np.median(values)),
                "std_difference": float(values.std())}
    for threshold in thresholds:
        features[f"percentage_changed_gt_{threshold:g}"] = float((values > threshold).mean() * 100)
    return features

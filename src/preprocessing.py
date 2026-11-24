"""Configurable, non-destructive classical image preprocessing."""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from .config import PreprocessingConfig
from .image_utils import validate_image


@dataclass(frozen=True)
class MorphologyConfig:
    """Optional binary-mask operations; zero iterations disables each operation."""

    kernel_size: int = 3
    opening_iterations: int = 0
    closing_iterations: int = 0
    erosion_iterations: int = 0
    dilation_iterations: int = 0


@dataclass(frozen=True)
class SegmentationConfig:
    """Explicit choices for exploratory classical foreground segmentation.

    ``dark_foreground`` is appropriate only while dark seeds are the intended
    foreground. These settings do not define a biological germination label.
    """

    method: str = "otsu"  # fixed, otsu, or adaptive
    fixed_threshold: int = 100
    adaptive_block_size: int = 51
    adaptive_c: float = 5.0
    dark_foreground: bool = True
    use_background_removal: bool = False
    background_kernel_size: int = 101
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)


def preprocess_for_analysis(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Return an analysis copy while leaving the caller's original image unchanged."""
    validate_image(image)
    result = image.copy()
    if config.roi:
        x, y, width, height = config.roi
        result = result[y:y + height, x:x + width]
    if config.grayscale and result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if config.gaussian_kernel:
        result = cv2.GaussianBlur(result, config.gaussian_kernel, 0)
    if config.median_kernel:
        result = cv2.medianBlur(result, config.median_kernel)
    if config.normalize_contrast:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    """Return a grayscale copy of a BGR image."""
    validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def reduce_noise(gray: np.ndarray, gaussian_kernel: tuple[int, int] | None = None,
                 median_kernel: int | None = None) -> np.ndarray:
    """Apply only explicitly requested Gaussian and/or median smoothing."""
    validate_image(gray)
    if gray.ndim != 2:
        raise ValueError("Noise reduction expects a grayscale image.")
    result = gray.copy()
    if gaussian_kernel:
        result = cv2.GaussianBlur(result, gaussian_kernel, 0)
    if median_kernel:
        result = cv2.medianBlur(result, median_kernel)
    return result


def estimate_background(gray: np.ndarray, kernel_size: int = 101) -> np.ndarray:
    """Estimate smooth illumination with a large median filter; kernel must be odd."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("background kernel_size must be odd and at least 3.")
    validate_image(gray)
    if gray.ndim != 2:
        raise ValueError("Background estimation expects grayscale input.")
    return cv2.medianBlur(gray, kernel_size)


def remove_background(gray: np.ndarray, kernel_size: int = 101) -> np.ndarray:
    """Highlight local dark/light deviations from a smooth estimated background."""
    background = estimate_background(gray, kernel_size)
    return cv2.absdiff(background, gray)


def threshold_image(gray: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Create a binary mask using fixed, Otsu, or adaptive thresholding."""
    validate_image(gray)
    if gray.ndim != 2:
        raise ValueError("Thresholding expects grayscale input.")
    threshold_type = cv2.THRESH_BINARY_INV if config.dark_foreground else cv2.THRESH_BINARY
    if config.method == "fixed":
        _, mask = cv2.threshold(gray, config.fixed_threshold, 255, threshold_type)
    elif config.method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif config.method == "adaptive":
        block = config.adaptive_block_size
        if block < 3 or block % 2 == 0:
            raise ValueError("adaptive_block_size must be odd and at least 3.")
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     threshold_type, block, config.adaptive_c)
    else:
        raise ValueError("method must be 'fixed', 'otsu', or 'adaptive'.")
    validate_binary_mask(mask)
    return mask


def validate_binary_mask(mask: np.ndarray) -> None:
    """Ensure a mask is uint8 and contains only background (0) and foreground (255)."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Mask must be a two-dimensional uint8 array.")
    if not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError("Mask values must be 0 or 255.")


def apply_morphology(mask: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    """Apply enabled binary-mask operations in a documented, fixed order.

    Opening, closing, erosion, then dilation are applied. Defaults do nothing,
    because aggressive morphology can remove thin candidate radicle structures.
    """
    validate_binary_mask(mask)
    if config.kernel_size < 1 or config.kernel_size % 2 == 0:
        raise ValueError("Morphology kernel_size must be positive and odd.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size))
    result = mask.copy()
    if config.opening_iterations:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=config.opening_iterations)
    if config.closing_iterations:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=config.closing_iterations)
    if config.erosion_iterations:
        result = cv2.erode(result, kernel, iterations=config.erosion_iterations)
    if config.dilation_iterations:
        result = cv2.dilate(result, kernel, iterations=config.dilation_iterations)
    validate_binary_mask(result)
    return result


def create_foreground_mask(image: np.ndarray, config: SegmentationConfig,
                           gaussian_kernel: tuple[int, int] | None = None,
                           median_kernel: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the threshold mask and optional morphologically cleaned foreground mask."""
    gray = reduce_noise(convert_to_gray(image), gaussian_kernel, median_kernel)
    threshold_input = remove_background(gray, config.background_kernel_size) if config.use_background_removal else gray
    mask = threshold_image(threshold_input, config)
    return mask, apply_morphology(mask, config.morphology)

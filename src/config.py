"""Project paths and explicit configuration defaults."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Locations relative to the repository root; raw files are read-only inputs."""

    root: Path = PROJECT_ROOT

    @property
    def raw_images(self) -> Path:
        return self.root / "data" / "raw" / "images"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Optional operations. Defaults preserve the image except grayscale conversion."""

    grayscale: bool = True
    gaussian_kernel: Optional[Tuple[int, int]] = None
    median_kernel: Optional[int] = None
    normalize_contrast: bool = False
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height


@dataclass(frozen=True)
class ChangeConfig:
    """Parameters for global consecutive-frame difference features.

    ``thresholds`` are pixel-intensity differences on an 8-bit grayscale scale.
    They are analysis choices, not definitions of germination.
    """

    thresholds: Tuple[float, ...] = (10.0, 25.0, 50.0)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

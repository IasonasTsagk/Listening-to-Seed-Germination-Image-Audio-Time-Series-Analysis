"""Central, conservative configuration for exploratory Audio Phase 1."""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class AudioPhase1Config:
    audio_root: Path = Path(os.environ.get("SEED_GERMINATION_AUDIO_ROOT", Path(__file__).resolve().parents[1] / "data/raw/audio"))
    analysis_start: str = "2026-07-13 00:00:00"
    analysis_end: str = "2026-07-16 23:59:59"
    envelope_window_seconds: float = 0.025
    envelope_hop_seconds: float = 0.010
    envelope_smoothing_frames: int = 3
    peak_min_distance_seconds: float = 0.10
    peak_width_seconds: float | None = None
    channel_strategy: str = "inspect_then_mean_mono"
    spectrogram_nperseg: int = 2048
    spectrogram_noverlap: int = 1536
    boundary_seconds: float = 0.05

    def validate_root(self) -> Path:
        if not self.audio_root.is_dir():
            raise FileNotFoundError(f"Audio root is not mounted or does not exist: {self.audio_root}")
        return self.audio_root

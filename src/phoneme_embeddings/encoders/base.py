"""Contract shared by audio encoders."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..audio import Audio


@dataclass(frozen=True)
class Frames:
    vectors: np.ndarray     # [T, D]
    intervals: np.ndarray   # [T, 2], start and end in seconds

    def __post_init__(self):
        if self.vectors.ndim != 2 or min(self.vectors.shape) < 1:
            raise ValueError("Expected a nonempty [T, D] embedding matrix")
        if self.intervals.shape != (len(self.vectors), 2):
            raise ValueError("Expected one interval per vector")
        if not np.isfinite(self.vectors).all() or not np.isfinite(self.intervals).all():
            raise ValueError("Encoder returned non-finite values")
        starts, ends = self.intervals.T
        if np.any(starts < 0) or np.any(ends <= starts) or np.any(np.diff(starts) < 0):
            raise ValueError("Expected ordered intervals with positive duration")


class Encoder(Protocol):
    def encode(self, audio: Audio) -> Frames: ...

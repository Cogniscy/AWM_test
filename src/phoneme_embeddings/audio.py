"""Read integer PCM WAV while preserving the original sample rate."""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # [samples, channels], float32
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def read_wav(path: Path) -> Audio:
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, count, *_ = wav.getparams()
        raw = wav.readframes(count)
    if width == 1:
        samples = np.frombuffer(raw, dtype="u1").astype(np.int32) - 128
    elif width in (2, 4):
        samples = np.frombuffer(raw, dtype=f"<i{width}")
    elif width == 3:
        b = np.frombuffer(raw, dtype="u1").reshape(-1, 3).astype(np.int32)
        unsigned = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        samples = (unsigned ^ 0x800000) - 0x800000
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    if not count or len(samples) != count * channels:
        raise ValueError(f"Empty or truncated WAV: {path}")
    return Audio(samples.reshape(-1, channels).astype(np.float32)
                 / np.float32(2 ** (8 * width - 1)), rate)

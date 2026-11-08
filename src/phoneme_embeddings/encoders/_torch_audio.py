"""Audio helpers for the legacy Torch adapters."""

import numpy as np

from .base import Frames

SAMPLE_RATE = 16000


def waveform_16k(audio):
    import torch
    from torchaudio.functional import resample

    signal = torch.from_numpy(audio.samples[:, 0].copy())
    return resample(signal, audio.sample_rate, SAMPLE_RATE)


def fixed_hop_frames(vectors, hop_samples, window_samples=None):
    starts = np.arange(len(vectors), dtype=np.float64) * hop_samples
    ends = starts + (window_samples if window_samples is not None else hop_samples)
    return Frames(vectors, np.column_stack((starts, ends)) / SAMPLE_RATE)

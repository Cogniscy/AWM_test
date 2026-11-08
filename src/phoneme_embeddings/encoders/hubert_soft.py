"""HuBERT-Soft adapter with the thesis frame grid for comparison."""

from pathlib import Path

import numpy as np

from ..audio import Audio
from .base import Frames


class HubertSoftEncoder:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"HuBERT-Soft checkpoint not found: {checkpoint}")
        import torch
        from torchaudio.functional import resample

        from .hubert_model import hubert_soft

        self.torch = torch
        self.resample = resample
        self.device = torch.device(device)
        self.model = hubert_soft(checkpoint).to(self.device)
        self.model.requires_grad_(False)

    def encode(self, audio: Audio) -> Frames:
        signal = self.torch.from_numpy(audio.samples[:, 0].copy())
        signal = self.resample(signal, audio.sample_rate, 16000)
        with self.torch.inference_mode():
            vectors = self.model.units(signal[None, None].to(self.device))[0]
            vectors = vectors.cpu().numpy()
        # Match the old extraction script's uniform grid over resampled audio.
        step = (len(signal) / 16000) / len(vectors)
        edges = np.arange(len(vectors) + 1, dtype=np.float64) * step
        return Frames(vectors, np.column_stack((edges[:-1], edges[1:])))

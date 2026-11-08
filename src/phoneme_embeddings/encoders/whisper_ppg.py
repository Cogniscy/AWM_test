"""Extract the Whisper audio encoder states used by the thesis."""

from pathlib import Path

from ._torch_audio import SAMPLE_RATE, fixed_hop_frames, waveform_16k


class WhisperPPGEncoder:
    def __init__(self, checkpoint, device="cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"WhisperPPG checkpoint not found: {checkpoint}")
        import torch
        from librosa.filters import mel
        from ._whisper_model import AudioEncoder

        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        dims = state["dims"]
        self.model = AudioEncoder(*(dims[f"n_{name}"] for name in (
            "mels", "audio_ctx", "audio_state", "audio_head", "audio_layer",
        )))
        prefix = "encoder."
        self.model.load_state_dict({
            k[len(prefix):]: value for k, value in state["model_state_dict"].items()
            if k.startswith(prefix)
        })
        self.model.to(self.device).eval().requires_grad_(False)
        self.filters = torch.from_numpy(mel(sr=SAMPLE_RATE, n_fft=400, n_mels=dims["n_mels"]))
        self.max_samples = dims["n_audio_ctx"] * 320

    def encode(self, audio):
        import torch
        import torch.nn.functional as F

        wave = waveform_16k(audio)
        count = len(wave) // 320
        if len(wave) > self.max_samples:
            raise ValueError("WhisperPPG input exceeds the checkpoint audio context; split the WAV first")
        if not count:
            raise ValueError("WhisperPPG needs at least one complete 20 ms frame")
        with torch.inference_mode():
            wave = F.pad(wave, (0, self.max_samples - len(wave)))
            stft = torch.stft(wave, 400, 160, window=torch.hann_window(400), return_complex=True)
            mel = (self.filters @ stft[..., :-1].abs().square()).clamp_min(1e-10).log10()
            mel = (torch.maximum(mel, mel.max() - 8.0) + 4.0) / 4.0
            vectors = self.model(mel[None].to(self.device))[0, :count].float().cpu().numpy()
        return fixed_hop_frames(vectors, 320)

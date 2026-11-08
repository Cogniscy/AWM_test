"""Fairseq ContentVec with the layer and projection from the thesis."""

from pathlib import Path

from ._torch_audio import fixed_hop_frames, waveform_16k


class ContentVecEncoder:
    def __init__(self, checkpoint, layer, project, device="cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ContentVec checkpoint not found: {checkpoint}")
        import torch
        from fairseq import checkpoint_utils

        self.device = torch.device(device)
        self.layer = layer
        self.project = project
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        self.model = models[0].to(self.device).eval().requires_grad_(False)
        self.hop = 1
        self.window = 1
        for layer in self.model.feature_extractor.conv_layers:
            conv = next(module for module in layer.modules() if isinstance(module, torch.nn.Conv1d))
            self.window += (conv.kernel_size[0] - 1) * conv.dilation[0] * self.hop
            self.hop *= conv.stride[0]

    def encode(self, audio):
        import torch

        source = waveform_16k(audio).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vectors, _ = self.model.extract_features(
                source=source, padding_mask=torch.zeros_like(source, dtype=torch.bool),
                output_layer=self.layer,
            )
            if self.project:
                vectors = self.model.final_proj(vectors)
        return fixed_hop_frames(vectors[0].float().cpu().numpy(), self.hop, self.window)

"""Extract the raw SylBoost features returned by the thesis adapter."""

from pathlib import Path

from ._torch_audio import fixed_hop_frames, waveform_16k


class SyllableLMEncoder:
    def __init__(self, checkpoint, device="cpu", layer=-2):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SylBoost checkpoint not found: {checkpoint}")
        import torch
        from ._syllablelm.config import Modality, d2v2_config
        from ._syllablelm.data2vec.models.data2vec2 import Data2VecMultiModel

        self.device = torch.device(device)
        self.layer = layer
        self.model = Data2VecMultiModel(d2v2_config, [Modality.AUDIO])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_seg"]
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        self.model.to(self.device).eval().requires_grad_(False)

    def encode(self, audio):
        import torch

        signal = waveform_16k(audio).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self.model(signal, mode=None, mask=False, features_only=True,
                                remove_extra_tokens=True, out_layer=self.layer)
            vectors = result["x"][0].float().cpu().numpy()
        return fixed_hop_frames(vectors, 320, 400)

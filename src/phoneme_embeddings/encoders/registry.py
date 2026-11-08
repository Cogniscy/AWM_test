"""Resolve adapters lazily so each model can have its own dependencies."""

from importlib import import_module

from .base import Encoder

ADAPTERS = {
    "hubert_soft": (".hubert_soft", "HubertSoftEncoder"),
    "contentvec_256_l9": (".contentvec", "ContentVecEncoder"),
    "contentvec_768_l12": (".contentvec", "ContentVecEncoder"),
    "whisper_ppg": (".whisper_ppg", "WhisperPPGEncoder"),
    "syllablelm": (".syllablelm", "SyllableLMEncoder"),
}


def create_encoder(name: str, **options) -> Encoder:
    if name not in ADAPTERS:
        raise ValueError(f"Unknown encoder {name!r}; available: {', '.join(sorted(ADAPTERS))}")
    module, cls = ADAPTERS[name]
    return getattr(import_module(module, __package__), cls)(**options)

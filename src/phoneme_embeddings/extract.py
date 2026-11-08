"""Extract and save frame and segment embeddings in one pass."""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .alignment import frame_metadata, normalize, segment_means
from .audio import read_wav
from .encoders.base import Encoder, Frames
from .encoders.registry import create_encoder
from .segmentation import read_seg_b2


def save_mode(directory: Path, mode: str, vectors: np.ndarray, metadata):
    np.save(directory / f"{mode}.npy", vectors, allow_pickle=False)
    with (directory / f"{mode}.jsonl").open("w", encoding="utf-8") as handle:
        for record in metadata:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract(wav: Path, segmentation: Path, out: Path, config: dict,
            *, on_stage: Callable[[str], None] | None = None,
            encoder_factory: Callable[..., Encoder] = create_encoder):
    def report(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    modes = config["modes"]
    if not isinstance(modes, list) or not modes or any(
        mode not in ("frame", "segment_mean") for mode in modes
    ) or len(modes) != len(set(modes)):
        raise ValueError("Specify frame, segment_mean, or both exactly once")
    normalization = config.get("normalization", "l2")
    if normalization not in ("none", "l2"):
        raise ValueError("normalization must be none or l2")
    threshold = float(config.get("overlap_threshold_sec", 0.001))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("overlap_threshold_sec must be finite and nonnegative")
    options = dict(config["encoder"])
    name = options.pop("name")
    if "checkpoint" in options:
        options["checkpoint"] = str(Path(options["checkpoint"]).expanduser().resolve())

    report("read")
    audio = read_wav(wav)
    segments = read_seg_b2(segmentation, audio.duration)
    report("load_model")
    encoder = encoder_factory(name, **options)
    report("encode")
    raw = encoder.encode(audio)
    frames = Frames(normalize(raw.vectors.astype(np.float32), normalization), raw.intervals)

    out.mkdir(parents=True, exist_ok=False)
    if "frame" in modes:
        report("frame")
        save_mode(out, "frame", frames.vectors, frame_metadata(frames.intervals, segments, threshold))
    if "segment_mean" in modes:
        report("segment_mean")
        save_mode(out, "segment_mean", *segment_means(frames, segments, normalization))
    report("metadata")
    run = {
        "wav": str(wav.resolve()), "segmentation": str(segmentation.resolve()),
        "encoder": {"name": name, **options}, "modes": modes,
        "normalization": normalization, "overlap_threshold_sec": threshold,
        "frame_count": len(frames.vectors), "dimension": frames.vectors.shape[1],
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--seg", required=True, type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", type=Path, help="New directory for this run")
    output.add_argument("--out-root", type=Path, help="Write to ROOT/CONFIG_NAME/WAV_NAME")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        out = args.out
        if out is None:
            out = args.out_root / args.config.stem / args.wav.stem
        extract(args.wav, args.seg, out, config)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        parser.exit(1, f"Extraction failed: {error}\n")


if __name__ == "__main__":
    main()

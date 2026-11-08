"""Read CORPRES .seg_B2 boundaries in seconds."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    id: int
    label: str
    start: float
    end: float


def read_seg_b2(path: Path, audio_duration: float) -> list[Segment]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251").removeprefix("п»ї")
    lines = [line.strip() for line in text.splitlines()]
    first = lines.index("[PARAMETERS]") + 1
    last = lines.index("[LABELS]")
    params = dict(line.split("=", 1) for line in lines[first:last] if line)
    byte_rate = (int(params["SAMPLING_FREQ"]) * int(params["BYTE_PER_SAMPLE"])
                 * int(params["N_CHANNEL"]))
    if byte_rate <= 0:
        raise ValueError("Invalid byte rate in .seg_B2")
    boundaries = []
    for line in lines[last + 1:]:
        if line.startswith("["):
            break
        if not line:
            continue
        offset, _, label = line.split(",", 2)
        byte_offset = float(offset)
        if not 0 <= byte_offset <= (audio_duration + 0.001) * byte_rate:
            raise ValueError(f"Boundary outside the WAV: {offset}")
        time = min(byte_offset / byte_rate, audio_duration)
        if boundaries and time < boundaries[-1][0]:
            raise ValueError("Unsorted .seg_B2 boundaries")
        boundaries.append((time, label))
    if not boundaries:
        raise ValueError(f"No boundaries in {path}")
    if boundaries[0][0] > 0:
        boundaries.insert(0, (0.0, "sil"))
    # A final empty marker closes the labeled track. Earlier empty intervals
    # are pauses, and a final named interval extends to the end of the WAV.
    if boundaries[-1][1]:
        boundaries.append((audio_duration, ""))
    return [
        Segment(i, label or "sil", start, end)
        for i, ((start, label), (end, _)) in enumerate(zip(boundaries, boundaries[1:]))
        if end > start
    ]

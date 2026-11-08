"""Align encoder output to the source segmentation."""

from bisect import bisect_right

import numpy as np

from .encoders.base import Frames
from .segmentation import Segment


def normalize(vectors: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return vectors
    if method != "l2":
        raise ValueError(f"Unknown normalization: {method}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1)


def frame_metadata(intervals: np.ndarray, segments: list[Segment], threshold: float):
    ends = [segment.end for segment in segments]
    for index, (start, end) in enumerate(intervals.tolist()):
        first = bisect_right(ends, start)
        candidates = []
        for i in range(first, len(segments)):
            if segments[i].start >= end:
                break
            candidates.append(segments[i])
        third = (end - start) / 3
        edges = (start, start + third, start + 2 * third, end)
        labels = []
        for left, right in zip(edges, edges[1:]):
            label, best = "sil", threshold
            for segment in candidates:
                overlap = min(right, segment.end) - max(left, segment.start)
                if overlap > best:
                    label, best = segment.label, overlap
            labels.append(label)
        contained = next((s.id for s in candidates if s.start <= start and end <= s.end), None)
        yield {
            "row": index, "start_sec": start, "end_sec": end,
            "labels": labels, "segment_id": contained,
        }


def segment_means(frames: Frames, segments: list[Segment], normalization: str):
    vectors = []
    metadata = []
    starts, ends = frames.intervals.T
    for segment in segments:
        first = np.searchsorted(starts, segment.start, side="left")
        last = np.searchsorted(starts, segment.end, side="left")
        selected = frames.vectors[first:last][ends[first:last] <= segment.end]
        row = None
        if len(selected):
            row = len(vectors)
            vectors.append(selected.mean(axis=0))
        metadata.append({
            "row": row, "segment_id": segment.id, "label": segment.label,
            "start_sec": segment.start, "end_sec": segment.end,
            "frame_count": len(selected),
        })
    matrix = np.asarray(vectors, dtype=frames.vectors.dtype).reshape(-1, frames.vectors.shape[1])
    return normalize(matrix, normalization), metadata

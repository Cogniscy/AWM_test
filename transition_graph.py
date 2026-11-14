"""Versioned feature transitions for CORPRES."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def content(graph):
    return (json.dumps(graph, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def version_id(graph):
    return hashlib.sha256(content(graph)).hexdigest()


def differences(labels, source, target):
    a, b = (labels[label]["features"] for label in (source, target))
    return {key: (a.get(key), b.get(key)) for key in sorted(a.keys() | b.keys())
            if a.get(key) != b.get(key)}


def transition(labels, source, target):
    diff = differences(labels, source, target)
    a, b = (labels[label]["features"] for label in (source, target))
    if source == target or a["type"] != b["type"] or a.keys() != b.keys() or len(diff) != 1:
        raise ValueError("Переход должен менять один признак внутри одного типа фонем.")
    feature, (before, after) = next(iter(diff.items()))
    return {"source": source, "target": target, "feature": feature,
            "from": before, "to": after, "enabled": True}


def initial_graph(configs):
    features = json.loads((configs / "corpres_features.json").read_text())
    pairs = json.loads((configs / "corpres_feature_pairs.json").read_text())
    labels = features["labels"]
    edges = [transition(labels, pair["source"], pair["target"]) for pair in pairs["pairs"]]
    return {"schema": 1, "labels": labels, "edges": edges,
            "classifier": features["classifier"],
            "split": json.loads((configs / pairs["split_config"]).read_text())}


def edit(graph, source, target, action):
    edges = {(e["source"], e["target"]): e.copy() for e in graph["edges"]}
    key = (source, target)
    reverse = (target, source)
    if action in ("disable", "reverse") and (key not in edges or not edges[key]["enabled"]):
        raise ValueError("Указанный переход не включён.")
    if action in ("disable", "reverse"):
        edges[key]["enabled"] = False
    if action in ("enable", "both"):
        edges[key] = transition(graph["labels"], *key)
    if action in ("reverse", "both"):
        edges[reverse] = transition(graph["labels"], *reverse)
    return {**graph, "edges": [edges[key] for key in sorted(edges)]}


def two_step_paths(edges):
    outgoing = {}
    for edge in edges:
        if edge["enabled"]:
            outgoing.setdefault(edge["source"], []).append(edge)
    paths = {}
    for source, first_edges in outgoing.items():
        for first in first_edges:
            for second in outgoing.get(first["target"], []):
                target = second["target"]
                if source != target:
                    paths.setdefault((source, target), []).append((first, second))
    return paths


def save_version(directory: Path, graph):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{version_id(graph)}.json"
    payload = content(graph)
    with NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError(f"Содержимое версии не совпадает с её идентификатором: {target.name}")
    finally:
        temporary.unlink()
    return target

"""Train and compare residual predictors on frozen CORPRES prototypes."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from phoneme_embeddings.classifier import CLASS_MAP, load_model
from transition_graph import content, transition, version_id


SPLITS = ("train", "validation", "test")


def operation(edge):
    return tuple(edge[key] for key in ("feature", "from", "to"))


def prepare(graph, pilot, held_pairs):
    manifest = json.loads((pilot / "metrics.json").read_text(encoding="utf-8"))
    if any(manifest[key] != graph["split"][key] for key in ("source_run", "speakers")):
        raise ValueError("Прототипы рассчитаны на другом прогоне или разбиении дикторов.")
    labels = manifest["labels"]
    indices = {label: index for index, label in enumerate(labels)}
    edges = [edge for edge in graph["edges"] if edge["enabled"]]
    if not edges:
        raise ValueError("В графе нет включённых переходов.")
    if len({(edge["source"], edge["target"]) for edge in edges}) != len(edges):
        raise ValueError("Граф содержит повторяющиеся переходы.")
    for edge in edges:
        expected = transition(graph["labels"], edge["source"], edge["target"])
        if edge != expected:
            raise ValueError("Признаки перехода не совпадают с таблицей графа.")
        if any(edge[key] not in indices for key in ("source", "target")):
            raise ValueError("Для одной из меток графа отсутствует прототип.")
    train_ids, test_ids = [], []
    held = {frozenset(pair) for pair in held_pairs}
    for index, edge in enumerate(edges):
        pair = frozenset((edge["source"], edge["target"]))
        (test_ids if pair in held else train_ids).append(index)
    if not held:
        test_ids = train_ids.copy()
    if not train_ids or not test_ids:
        raise ValueError("Для обучения и теста нужны непустые наборы переходов.")
    operations = list(dict.fromkeys(operation(edge) for edge in edges))
    train_ops = {operation(edges[index]) for index in train_ids}
    unseen = {operation(edges[index]) for index in test_ids} - train_ops
    if unseen:
        names = ", ".join(f"{feature} ({before} → {after})" for feature, before, after in sorted(unseen, key=str))
        raise ValueError(f"После исключения пар у этих операций нет примеров в обучении: {names}")
    condition = torch.eye(len(operations))[[operations.index(operation(edge)) for edge in edges]]
    batches = {}
    with np.load(pilot / "prototypes.npz", allow_pickle=False) as data:
        for split, ids in zip(SPLITS, (train_ids, train_ids, test_ids)):
            vectors = data[split]
            if vectors.shape != (len(labels), 256) or not np.isfinite(vectors).all():
                raise ValueError(f"Неверные прототипы для {split}.")
            chosen = [edges[index] for index in ids]
            source = torch.tensor(vectors[[indices[e["source"]] for e in chosen]], dtype=torch.float32)
            target = torch.tensor(vectors[[indices[e["target"]] for e in chosen]], dtype=torch.float32)
            batches[split] = source, target, condition[ids], chosen
    return batches, operations


class ResidualPredictor(nn.Module):
    def __init__(self, conditions, width, depth):
        super().__init__()
        layers = [nn.Linear(256 + conditions, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        self.delta = nn.Sequential(*layers, nn.Linear(width, 256))

    def forward(self, source, condition):
        return source + self.delta(torch.cat((source, condition), dim=1))


def train_model(train, validation, settings, conditioned, notify):
    x, y, c, _ = train
    vx, vy, vc, _ = validation
    if not conditioned:
        c, vc = torch.zeros_like(c), torch.zeros_like(vc)
    # Identical initial weights for paired runs with the same seed.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(settings["seed"])
        model = ResidualPredictor(c.shape[1], settings["width"], settings["depth"])
    optim = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        optim.zero_grad()
        loss = criterion(model(x, c), y)
        loss.backward()
        optim.step()
        with torch.no_grad():
            validation_loss = criterion(model(vx, vc), vy).item()
        if not np.isfinite(validation_loss):
            raise ValueError("Ошибка обучения стала бесконечной. Уменьши learning rate.")
        history.append({"epoch": epoch, "train_mse": loss.item(), "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == settings["epochs"]:
            notify(epoch, validation_loss)
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, history


def run_experiment(graph, batches, operations, settings, checkpoint, pilot, output, notify):
    mapping = CLASS_MAP[graph["classifier"]]
    indices = {label: int(index) for index, label in mapping.items()}
    classifier = load_model(checkpoint, len(mapping)).requires_grad_(False)
    x, y, condition, edges = batches["test"]
    targets = torch.tensor([indices[e["target"]] for e in edges])

    @torch.inference_mode()
    def score(prediction):
        actual = classifier(prediction)[1].argmax(dim=1)
        correct = actual == targets
        return {"mse": (prediction - y).square().mean().item(),
                "accuracy": correct.double().mean().item(),
                "correct": correct.sum().item(),
                "total": len(targets),
                "macro_accuracy": torch.stack([correct[targets == label].double().mean()
                                               for label in targets.unique()]).mean().item(),
                "predictions": [{**edge, "predicted": mapping[str(label)],
                                 "mse": error}
                                for edge, label, error in zip(
                                    edges, actual.tolist(), (prediction - y).square().mean(1).tolist())]}

    train_source, train_target, train_condition, _ = batches["train"]
    mean_deltas = torch.stack([(train_target - train_source)[train_condition[:, index].bool()].mean(0)
                               if train_condition[:, index].any() else torch.zeros_like(train_source[0])
                               for index in range(len(operations))])
    scores = {"identity": score(x), "mean_shift": score(x + condition @ mean_deltas),
              "oracle": score(y)}
    learned = {}
    for name in settings["models"]:
        conditioned = name == "conditioned"
        model, epoch, history = train_model(
            batches["train"], batches["validation"], settings, conditioned,
            lambda step, loss: notify(name, step, loss))
        with torch.inference_mode():
            result = score(model(x, condition if conditioned else torch.zeros_like(condition)))
        scores[name] = {**result, "epoch": epoch, "history": history,
                        "parameters": sum(p.numel() for p in model.parameters())}
        learned[name] = model
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "graph_version": version_id(graph), "settings": settings, "operations": operations,
        "classifier": graph["classifier"], "checkpoint": str(checkpoint),
        "inputs_sha256": {
            "classifier": checkpoint_hash,
            "prototypes": hashlib.sha256((pilot / "prototypes.npz").read_bytes()).hexdigest(),
        },
        "prototype_directory": str(pilot), "speakers": graph["split"]["speakers"],
        "splits": {split: [dict(edge) for edge in batches[split][3]] for split in SPLITS},
        "metrics": scores,
    }
    output.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    with TemporaryDirectory(dir=output) as staging:
        pending = Path(staging) / name
        pending.mkdir()
        (pending / "graph.json").write_bytes(content(graph))
        (pending / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        for key, model in learned.items():
            torch.save(model.state_dict(), pending / f"{key}.pth")
        pending.rename(output / name)
    return output / name, result

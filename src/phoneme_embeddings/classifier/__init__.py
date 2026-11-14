"""Evaluate the thesis baseline on saved 256-dimensional HuBERT-Soft frames."""

import json
from pathlib import Path

import torch
from torch import nn


CLASS_MAP = json.loads(Path(__file__).with_name("classifier_classes.json").read_text())


class ClassificationModel(nn.Module):
    def __init__(self, classes, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(256, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.final_start = nn.Linear(64, classes)
        self.final_center = nn.Linear(64, classes)
        self.final_end = nn.Linear(64, classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        for layer in (self.linear1, self.linear2, self.linear3, self.linear4):
            embeddings = self.dropout(self.relu(layer(embeddings)))
        return tuple(head(embeddings) for head in
                     (self.final_start, self.final_center, self.final_end))


def load_model(checkpoint, class_count):
    allowed = [(ClassificationModel, "DatasetsModels.ClassificationModel"),
               nn.Linear, nn.ReLU, nn.Dropout, set]
    # The original checkpoints serialize the module itself under DatasetsModels.
    with torch.serialization.safe_globals(allowed):
        model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(model, dict) and model.get("pipeline") in {"triplet", "center"}:
        state = model
        model = ClassificationModel(class_count)
        model.load_state_dict(state["state_dict"])
    if (not isinstance(model, ClassificationModel) or model.linear1.in_features != 256
            or model.final_center.out_features != class_count):
        raise ValueError("Checkpoint не соответствует выбранной модели.")
    return model.eval()


def evaluate(model_id, checkpoint, test_path):
    mapping = CLASS_MAP[model_id]
    labels = [mapping[str(index)] for index in range(len(mapping))]
    indices = {label: index for index, label in enumerate(labels)}
    model = load_model(checkpoint, len(labels))
    rows = torch.load(test_path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ожидается непустой список кадров в файле .pt.")

    count = len(labels)
    confusion = torch.zeros(count * count, dtype=torch.int64)
    used = 0
    for offset in range(0, len(rows), 4096):
        embeddings, targets = [], []
        for row in rows[offset:offset + 4096]:
            if not isinstance(row, dict):
                raise ValueError("Каждый кадр должен содержать embedding и label.")
            frame, triplet = row.get("embedding"), row.get("label")
            if (not isinstance(frame, torch.Tensor) or frame.shape != (256,)
                    or not isinstance(triplet, (list, tuple)) or len(triplet) != 3
                    or not all(isinstance(label, str) for label in triplet)):
                raise ValueError("Ожидаются вектор из 256 признаков и три фонемные метки.")
            if all(label in indices for label in triplet):
                embeddings.append(frame)
                targets.append(indices[triplet[1]])
        if embeddings:
            batch = torch.stack(embeddings).float()
            if not torch.isfinite(batch).all():
                raise ValueError("В эмбеддингах есть NaN или бесконечные значения.")
            with torch.inference_mode():
                predicted = model(batch)[1].argmax(dim=1)
            confusion += torch.bincount(torch.tensor(targets) * count + predicted,
                                        minlength=count * count)
            used += len(targets)
    if not used:
        raise ValueError("В файле нет кадров с метками из словаря модели.")

    matrix = confusion.reshape(count, count)
    support, predicted, correct = matrix.sum(1), matrix.sum(0), matrix.diag()
    observed = support > 0
    union = support + predicted > 0
    return {
        "model": model_id,
        "checkpoint": str(checkpoint),
        "test_embeddings": str(test_path),
        "frames": used,
        "excluded_frames": len(rows) - used,
        "accuracy": correct.sum().item() / used,
        "balanced_accuracy": (correct[observed].double() / support[observed]).mean().item(),
        "macro_f1": (2 * correct[union].double() / (support + predicted)[union]).mean().item(),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

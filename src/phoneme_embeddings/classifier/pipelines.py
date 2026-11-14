"""Composable training pipelines for saved HuBERT-Soft embeddings."""

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from . import CLASS_MAP, ClassificationModel


def frame_data(path, labels, limit):
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list):
        raise ValueError("Ожидается список кадров в файле .pt.")
    frames, targets = [], []
    for row in rows:
        frame, triplet = row["embedding"], row["label"]
        if frame.shape != (256,) or len(triplet) != 3:
            raise ValueError("Ожидаются 256 признаков и три метки на кадр.")
        if all(label in labels for label in triplet):
            frames.append(frame)
            targets.append([labels[label] for label in triplet])
            if limit and len(frames) == limit:
                break
    if not frames:
        raise ValueError(f"Нет подходящих кадров в {path}")
    return TensorDataset(torch.stack(frames).float(), torch.tensor(targets))


def sequence_data(path, labels, limit):
    if not path.is_dir():
        raise ValueError(f"Для CTC укажи папку с парами .npy и .txt: {path}")
    sequences = []
    for annotation in sorted(path.rglob("*_no_av_ph.txt")):
        embedding = annotation.with_name(annotation.name.removesuffix("_no_av_ph.txt") + "_no_av_embs.npy")
        triplets = [line.split("_") for line in annotation.read_text().splitlines()]
        x = np.load(embedding, allow_pickle=False)
        if x.shape != (len(triplets), 256) or any(len(row) != 3 for row in triplets):
            raise ValueError(f"Несовместимые эмбеддинги или метки: {annotation}")
        centers = [row[1] for row in triplets]
        if not centers or any(label not in labels for label in centers):
            continue
        target = [labels[label] + 1 for label, _ in groupby(centers)]
        sequences.append((torch.from_numpy(x).float(), torch.tensor(target)))
        if limit and len(sequences) == limit:
            break
    if not sequences:
        raise ValueError(f"Нет высказываний с метками из словаря в {path}")
    return sequences


def sequence_batch(batch):
    frames, targets = zip(*batch)
    return (pad_sequence(frames, batch_first=True), torch.cat(targets),
            torch.tensor([len(x) for x in frames]), torch.tensor([len(y) for y in targets]))


class CTCModel(nn.Module):
    def __init__(self, classes, dropout):
        super().__init__()
        self.encoder = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=dropout)
        self.output = nn.Linear(256, classes + 1)

    def forward(self, frames, lengths):
        packed = pack_padded_sequence(frames, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True)
        return self.output(padded)


def triplet_loss(model, batch):
    x, labels = batch
    return sum(nn.functional.cross_entropy(logits, labels[:, position])
               for position, logits in enumerate(model(x)))


def center_loss(model, batch):
    x, labels = batch
    return nn.functional.cross_entropy(model(x)[1], labels[:, 1])


def ctc_loss(model, batch):
    frames, target, lengths, target_lengths = batch
    logits = model(frames, lengths).log_softmax(-1).transpose(0, 1)
    return nn.functional.ctc_loss(logits, target, lengths, target_lengths)


@dataclass(frozen=True)
class Pipeline:
    data: object
    model: object
    loss: object
    collate: object = None


PIPELINES = {
    "triplet": Pipeline(frame_data, ClassificationModel, triplet_loss),
    "center": Pipeline(frame_data, ClassificationModel, center_loss),
    "ctc": Pipeline(sequence_data, CTCModel, ctc_loss, sequence_batch),
}


@dataclass(frozen=True)
class TrainingConfig:
    pipeline: str
    vocabulary: str
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout: float = 0.1
    weight_decay: float = 0.0001
    optimizer: str = "AdamW"
    schedule: str = "cosine"
    seed: int = 42
    limit: int = 0


def run_training(config, train_path, validation_path, output, on_epoch=None):
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Укажи разные источники обучения и валидации.")
    if (config.pipeline == "ctc" and
            (train_path.resolve() in validation_path.resolve().parents
             or validation_path.resolve() in train_path.resolve().parents)):
        raise ValueError("Папки обучения и валидации не должны быть вложены друг в друга.")
    pipeline = PIPELINES[config.pipeline]
    mapping = CLASS_MAP[config.vocabulary]
    names = [mapping[str(i)] for i in range(len(mapping))]
    labels = {name: i for i, name in enumerate(names)}
    train = pipeline.data(train_path, labels, config.limit)
    validation = pipeline.data(validation_path, labels, config.limit)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                              collate_fn=pipeline.collate, generator=generator)
    validation_loader = DataLoader(validation, batch_size=config.batch_size,
                                   collate_fn=pipeline.collate)
    directory = output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    history = []
    best_loss = float("inf")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = pipeline.model(len(labels), config.dropout)
        optimizer = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[config.optimizer](
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        schedulers = {
            "cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs),
            "step": lambda: torch.optim.lr_scheduler.StepLR(optimizer, max(1, config.epochs // 3)),
            "constant": lambda: None,
        }
        scheduler = schedulers[config.schedule]()
        for epoch in range(1, config.epochs + 1):
            train_loss = run_epoch(pipeline, model, train_loader, optimizer)
            validation_loss = run_epoch(pipeline, model, validation_loader)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            if validation_loss < best_loss:
                best_loss = validation_loss
                temporary = directory / "best.tmp"
                torch.save({"pipeline": config.pipeline, "vocabulary": config.vocabulary,
                            "labels": names, "state_dict": model.state_dict(), "epoch": epoch},
                           temporary)
                temporary.replace(directory / "best.pth")
            report = {"config": asdict(config), "train": str(train_path),
                      "validation": str(validation_path), "train_examples": len(train),
                      "validation_examples": len(validation), "history": history}
            (directory / "history.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if scheduler:
                scheduler.step()
            if on_epoch:
                on_epoch(history)
    return directory, history


def run_epoch(pipeline, model, loader, optimizer=None):
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            loss = pipeline.loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Функция потерь вернула нечисловое значение.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = len(batch[0])
            total += loss.item() * size
            count += size
    return total / count

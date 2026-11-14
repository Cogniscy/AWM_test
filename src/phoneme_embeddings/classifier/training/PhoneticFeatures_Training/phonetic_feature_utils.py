import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset


FEATURE_NAMES = [
    "syl",
    "son",
    "cons",
    "cont",
    "delrel",
    "lat",
    "nas",
    "strid",
    "voi",
    "sg",
    "cg",
    "ant",
    "cor",
    "distr",
    "lab",
    "hi",
    "lo",
    "back",
    "round",
    "velaric",
    "tense",
    "long",
    "add",
    "other",
]


class PhonemeEmbeddingDataset(Dataset):
    def __init__(
        self,
        data,
        phoneme_list,
        corpres_to_ipa_symbol=None,
        panphon_features=None,
        triplet_labels=False,
        target_position=1,
    ):
        super().__init__()
        self.data = data
        self.triplet_labels = triplet_labels
        self.target_position = target_position
        self.phoneme_list = phoneme_list
        self.phoneme2idx = {ph: i for i, ph in enumerate(self.phoneme_list)}
        self.corpres_to_ipa_symbol = corpres_to_ipa_symbol
        self.panphon_features = panphon_features
        self.phoneme_feature_cache = {}

        if corpres_to_ipa_symbol is not None and panphon_features is not None:
            self._precompute_features()

    @staticmethod
    def _encode_features(vector):
        mapping = {-1: 0, 1: 1, 0: 2}
        return [mapping[v] for v in vector]

    def _precompute_features(self):
        for phoneme in self.phoneme_list:
            ipa_symbol = self.corpres_to_ipa_symbol(phoneme)
            panphon_vector = self.panphon_features(ipa_symbol)

            if panphon_vector:
                features = self._encode_features(panphon_vector[0].numeric())
            else:
                features = None

            self.phoneme_feature_cache[phoneme] = {
                "ipa": ipa_symbol,
                "features": features,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        embedding = sample["embedding"]
        phonemes = sample["label"] if self.triplet_labels else [sample["label"][self.target_position]]

        phoneme_indices = []
        ipa_symbols = []
        phonetic_features = []

        for phoneme in phonemes:
            phoneme_indices.append(self.phoneme2idx.get(phoneme, -1))
            cached = self.phoneme_feature_cache.get(phoneme)

            if cached is None or cached["features"] is None:
                ipa_symbols.append(phoneme)
                phonetic_features.append(None)
            else:
                ipa_symbols.append(cached["ipa"])
                phonetic_features.append(cached["features"])

        phoneme_labels = torch.tensor(phoneme_indices, dtype=torch.long)
        feature_tensor = None

        if phonetic_features and phonetic_features[0] is not None:
            feature_tensor = torch.tensor(phonetic_features, dtype=torch.long)
            if not self.triplet_labels:
                feature_tensor = feature_tensor[0]

        return embedding, phoneme_labels, phonemes, ipa_symbols, feature_tensor


def compute_class_weights(dataset, num_features):
    all_targets = [features for _, _, _, _, features in dataset if features is not None]
    all_targets = torch.stack(all_targets)
    weights_by_feature = {}

    for feature_idx in range(num_features):
        values = all_targets[:, feature_idx]
        counts = torch.bincount(values, minlength=int(values.max().item()) + 1).float()
        nonzero_mask = counts > 0
        present_classes = int(nonzero_mask.sum().item())
        total = counts.sum().item()

        weights = torch.ones_like(counts)
        weights[nonzero_mask] = total / (present_classes * counts[nonzero_mask])
        weights_by_feature[feature_idx] = weights

    return weights_by_feature


def compute_class_priors(dataset, num_features, num_classes_per_feature=None, ignore_index=None):
    all_targets = [features for _, _, _, _, features in dataset if features is not None]
    all_targets = torch.stack(all_targets)
    priors_by_feature = {}

    for feature_idx in range(num_features):
        values = all_targets[:, feature_idx]
        if ignore_index is not None:
            values = values[values != ignore_index]

        if num_classes_per_feature is not None:
            num_classes = int(num_classes_per_feature[feature_idx])
        elif values.numel() > 0:
            num_classes = int(values.max().item()) + 1
        else:
            raise ValueError(
                "Cannot infer num_classes for an empty feature slice. "
                "Pass num_classes_per_feature explicitly."
            )

        if ignore_index is not None:
            num_classes = num_classes_per_feature[feature_idx]

        counts = torch.bincount(values, minlength=num_classes).float()
        total = counts.sum()
        if total <= 0:
            priors = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)
        else:
            priors = counts / total

        priors_by_feature[feature_idx] = priors

    return priors_by_feature


def compute_resistance_bias(
    priors_by_feature,
    smoothing_alpha=1.0,
    epsilon=1e-3,
):
    bias_by_feature = {}

    for feature_idx, priors in priors_by_feature.items():
        scaled = priors.float().pow(smoothing_alpha)
        normalized = scaled / scaled.sum().clamp_min(torch.finfo(scaled.dtype).eps)
        bias_by_feature[feature_idx] = -torch.log(normalized + epsilon)

    return bias_by_feature


def build_resistance_bias_dict(
    dataset,
    num_features,
    num_classes_per_feature=None,
    ignore_index=None,
    smoothing_alpha=1.0,
    epsilon=1e-3,
):
    priors_by_feature = compute_class_priors(
        dataset,
        num_features=num_features,
        num_classes_per_feature=num_classes_per_feature,
        ignore_index=ignore_index,
    )
    return compute_resistance_bias(
        priors_by_feature,
        smoothing_alpha=smoothing_alpha,
        epsilon=epsilon,
    )
def _align_bias_to_num_classes(bias, num_classes):
    if bias.numel() == num_classes:
        return bias

    aligned = torch.zeros(num_classes, dtype=bias.dtype, device=bias.device)
    copy_size = min(num_classes, bias.numel())
    aligned[:copy_size] = bias[:copy_size]
    return aligned

def infer_num_classes(dataset):
    all_targets = [features for _, _, _, _, features in dataset if features is not None]
    all_targets = torch.stack(all_targets)
    return [int(all_targets[:, idx].max().item()) + 1 for idx in range(all_targets.shape[1])]


class PhonemeFeaturePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes_per_feature, active_heads=None, dropout=0.5):
        super().__init__()
        self.num_classes_per_feature = list(num_classes_per_feature)
        self.num_features = len(self.num_classes_per_feature)
        self.active_heads = sorted(active_heads or list(range(self.num_features)))

        invalid_heads = [idx for idx in self.active_heads if idx < 0 or idx >= self.num_features]
        if invalid_heads:
            raise ValueError(f"Active heads out of range: {invalid_heads}")

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.num_classes_per_feature[idx]) for idx in range(self.num_features)]
        )

    def forward(self, x):
        shared = self.backbone(x)
        return [head(shared) if idx in self.active_heads else None for idx, head in enumerate(self.heads)]


def compute_loss(outputs, targets, active_heads, criterion_dict, resistance_bias_dict=None):
    total_loss = 0
    losses_per_feature = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        target = targets[:, feature_idx]
        if resistance_bias_dict is not None and feature_idx in resistance_bias_dict:
            #logits = logits - resistance_bias_dict[feature_idx].to(logits.device)
            bias = resistance_bias_dict[feature_idx].to(logits.device)
            bias = _align_bias_to_num_classes(bias, logits.shape[1])
            logits = logits - bias
        loss = criterion_dict[feature_idx](logits, target)
        total_loss += loss
        losses_per_feature[feature_idx] = loss.item()

    return total_loss, losses_per_feature


def compute_accuracy(outputs, targets, active_heads):
    accuracies = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        predictions = torch.argmax(logits, dim=1)
        accuracies[feature_idx] = (predictions == targets[:, feature_idx]).float().mean().item()

    return sum(accuracies.values()) / len(accuracies), accuracies

def _filter_ignored_targets(preds, true, ignore_index=None):
    if ignore_index is None:
        return preds, true

    valid_mask = true != ignore_index
    if valid_mask.sum().item() == 0:
        return None, None

    return preds[valid_mask], true[valid_mask]


def _compute_balanced_accuracy_from_predictions(preds, true):
    recalls = []
    for cls in torch.unique(true):
        cls_mask = true == cls
        recall = (preds[cls_mask] == true[cls_mask]).float().mean().item()
        recalls.append(recall)

    return float(sum(recalls) / len(recalls))


def _mean_without_nan(values):
    valid_values = [value for value in values if not math.isnan(value)]
    if not valid_values:
        return float("nan")
    return float(sum(valid_values) / len(valid_values))


def compute_balanced_accuracy(outputs, targets, active_heads, ignore_index=None):
    per_feature_balanced_acc = {}

    for feature_idx in active_heads:
        logits = outputs[feature_idx]
        preds = torch.argmax(logits, dim=1)
        true = targets[:, feature_idx]

        preds, true = _filter_ignored_targets(preds, true, ignore_index=ignore_index)
        if preds is None:
            per_feature_balanced_acc[feature_idx] = float("nan")
            continue

        per_feature_balanced_acc[feature_idx] = _compute_balanced_accuracy_from_predictions(preds, true)

    overall_balanced_acc = _mean_without_nan(per_feature_balanced_acc.values())

    return overall_balanced_acc, per_feature_balanced_acc



@torch.no_grad()
def evaluate(model, loader, device, active_heads, criterion_dict, ignore_index=None):
    model.eval()
    total_loss = 0.0
    all_preds = {feature_idx: [] for feature_idx in active_heads}
    all_targets = {feature_idx: [] for feature_idx in active_heads}

    for embeddings, _, _, _, features in loader:
        embeddings = embeddings.to(device)
        targets = features.to(device)
        outputs = model(embeddings)

        loss, _ = compute_loss(outputs, targets, active_heads, criterion_dict)
        total_loss += loss.item()

        for feature_idx in active_heads:
            preds = torch.argmax(outputs[feature_idx], dim=1)
            true = targets[:, feature_idx]
            preds, true = _filter_ignored_targets(preds, true, ignore_index=ignore_index)
            if preds is None:
                continue
            all_preds[feature_idx].append(preds.cpu())
            all_targets[feature_idx].append(true.cpu())

    per_feature_accuracy = {}
    per_feature_balanced_acc = {}
    for feature_idx in active_heads:
        if not all_targets[feature_idx]:
            per_feature_accuracy[feature_idx] = float("nan")
            per_feature_balanced_acc[feature_idx] = float("nan")
            continue

        preds = torch.cat(all_preds[feature_idx])
        true = torch.cat(all_targets[feature_idx])
        per_feature_accuracy[feature_idx] = (preds == true).float().mean().item()
        per_feature_balanced_acc[feature_idx] = _compute_balanced_accuracy_from_predictions(preds, true)

    return {
        "loss": total_loss / len(loader),
        "accuracy": _mean_without_nan(per_feature_accuracy.values()),
        "balanced_accuracy": _mean_without_nan(per_feature_balanced_acc.values()),
        "per_feature_acc": per_feature_accuracy,
        "per_feature_balanced_acc": per_feature_balanced_acc,
    }


def evaluate_before_training(model, loader, device, active_heads, criterion_dict, ignore_index=None):
    results = evaluate(
        model,
        loader,
        device,
        active_heads,
        criterion_dict,
        ignore_index=ignore_index,
    )
    print("\nBaseline before training:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    for feature_idx, value in results["per_feature_balanced_acc"].items():
        print(f"Feature {feature_idx} balanced acc: {value:.4f}, Accuracy: {results['per_feature_acc'][feature_idx]:.4f}")
    return results["loss"], results["balanced_accuracy"], results["per_feature_balanced_acc"]


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    active_heads,
    criterion_dict,
    epochs=10,
    eval_steps_per_epoch=4,
    save_model_path=None,
    ignore_index=None,
    resistance_bias_dict=None,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_feature_acc": {feature_idx: [] for feature_idx in active_heads},
        "val_feature_acc": {feature_idx: [] for feature_idx in active_heads},
        "train_accuracy": [],
        "val_accuracy": [],
        "train_feature_accuracy": {feature_idx: [] for feature_idx in active_heads},
        "val_feature_accuracy": {feature_idx: [] for feature_idx in active_heads},
        "steps": [],
    }
    best_val_loss = float("inf")
    step = 0
    eval_interval = max(1, len(train_loader) // max(1, eval_steps_per_epoch))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_preds = {feature_idx: [] for feature_idx in active_heads}
        epoch_targets = {feature_idx: [] for feature_idx in active_heads}
        last_val_loss = None

        for batch_idx, (embeddings, _, _, _, features) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            targets = features.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss, _ = compute_loss(
                outputs,
                targets,
                active_heads,
                criterion_dict,
                resistance_bias_dict=resistance_bias_dict,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            for feature_idx in active_heads:
                preds = torch.argmax(outputs[feature_idx], dim=1)
                true = targets[:, feature_idx]
                preds, true = _filter_ignored_targets(preds, true, ignore_index=ignore_index)
                if preds is None:
                    continue
                epoch_preds[feature_idx].append(preds.detach().cpu())
                epoch_targets[feature_idx].append(true.detach().cpu())

            step += 1
            should_eval = ((batch_idx + 1) % eval_interval == 0) or (batch_idx + 1 == len(train_loader))
            if not should_eval:
                continue

            train_metrics = evaluate(
                model,
                train_loader,
                device,
                active_heads,
                criterion_dict,
                ignore_index=ignore_index,
            )

            # train_loss = running_loss / (batch_idx + 1)
            # train_feature_accuracy = {}
            # train_feature_balanced_acc = {}
            # for feature_idx in active_heads:
            #     if not epoch_targets[feature_idx]:
            #         train_feature_accuracy[feature_idx] = float("nan")
            #         train_feature_balanced_acc[feature_idx] = float("nan")
            #         continue
            #
            #     preds = torch.cat(epoch_preds[feature_idx])
            #     true = torch.cat(epoch_targets[feature_idx])
            #     train_feature_accuracy[feature_idx] = (preds == true).float().mean().item()
            #     train_feature_balanced_acc[feature_idx] = _compute_balanced_accuracy_from_predictions(preds, true)
            #
            # train_accuracy = _mean_without_nan(train_feature_accuracy.values())
            # train_balanced_acc = _mean_without_nan(train_feature_balanced_acc.values())

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                active_heads,
                criterion_dict,
                ignore_index=ignore_index,
            )
            last_val_loss = val_metrics["loss"]

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["balanced_accuracy"])
            history["val_acc"].append(val_metrics["balanced_accuracy"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            for feature_idx in active_heads:
                history["train_feature_acc"][feature_idx].append(train_metrics["per_feature_balanced_acc"][feature_idx])
                history["val_feature_acc"][feature_idx].append(val_metrics["per_feature_balanced_acc"][feature_idx])
                history["train_feature_accuracy"][feature_idx].append(train_metrics["per_feature_balanced_acc"][feature_idx])
                history["val_feature_accuracy"][feature_idx].append(val_metrics["per_feature_acc"][feature_idx])
            history["steps"].append(step)

            print(
                f"Epoch {epoch + 1} | step {batch_idx + 1}/{len(train_loader)} | "
                f"train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
                f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} | val_bal_acc={val_metrics['balanced_accuracy']:.4f}"
            )

        if scheduler is not None:
            scheduler.step()

        if save_model_path is not None and last_val_loss is not None and last_val_loss < best_val_loss:
            best_val_loss = last_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_loss": best_val_loss,
                    "model_config": {
                        "input_dim": model.backbone[0].in_features,
                        "hidden_dim": model.backbone[0].out_features,
                        "num_classes_per_feature": model.num_classes_per_feature,
                        "active_heads": model.active_heads,
                        "ignore_index": ignore_index,
                    },
                },
                save_model_path,
            )
            print(f"  -> Best model saved at epoch {epoch + 1} with val_loss={best_val_loss:.4f}")

    return history


def plot_training(history, baseline_loss, baseline_acc):
    steps = history["steps"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(0, baseline_loss, label="baseline")
    plt.plot(steps, history["train_loss"], label="train")
    plt.plot(steps, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(0, baseline_acc, label="baseline")
    plt.plot(steps, history["train_acc"], label="train")
    plt.plot(steps, history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_feature_accuracy(history, baseline_feature_acc, feature_names, active_heads):
    for feature_idx in active_heads:
        if feature_idx not in baseline_feature_acc:
            continue

        plt.figure()
        plt.scatter(0, baseline_feature_acc[feature_idx], label="baseline")
        plt.plot(history["steps"], history["train_feature_acc"][feature_idx], label="train")
        plt.plot(history["steps"], history["val_feature_acc"][feature_idx], label="val")
        plt.title(f"Feature '{feature_names[feature_idx]}' accuracy")
        plt.xlabel("step")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()


@torch.no_grad()
def compute_confusion_matrices(model, loader, device, active_heads, ignore_index=None):
    model.eval()
    all_preds = {feature_idx: [] for feature_idx in active_heads}
    all_targets = {feature_idx: [] for feature_idx in active_heads}

    for embeddings, _, _, _, features in loader:
        embeddings = embeddings.to(device)
        targets = features.to(device)
        outputs = model(embeddings)

        for feature_idx in active_heads:
            predictions = torch.argmax(outputs[feature_idx], dim=1)
            true = targets[:, feature_idx]
            predictions, true = _filter_ignored_targets(predictions, true, ignore_index=ignore_index)
            if predictions is None:
                continue
            all_preds[feature_idx].extend(predictions.cpu().tolist())
            all_targets[feature_idx].extend(true.cpu().tolist())

    result = {}
    for feature_idx in active_heads:
        labels = sorted(set(all_targets[feature_idx]) | set(all_preds[feature_idx]))
        result[feature_idx] = {
            "matrix": confusion_matrix(all_targets[feature_idx], all_preds[feature_idx], labels=labels),
            "labels": labels,
        }

    return result


def _init_heatmap_axes(num_plots):
    rows = int(math.ceil(num_plots / 3))
    fig, axes = plt.subplots(ncols=3, nrows=rows, figsize=(18, 5 * rows))
    return fig, np.atleast_1d(axes).flatten()


def plot_confusion_matrices(confusion_matrices, feature_names):
    fig, axes = _init_heatmap_axes(len(confusion_matrices))

    for plot_idx, (feature_idx, payload) in enumerate(confusion_matrices.items()):
        sns.heatmap(
            payload["matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[plot_idx],
            cbar_kws={"shrink": 0.8},
            xticklabels=payload["labels"],
            yticklabels=payload["labels"],
        )
        axes[plot_idx].set_title(f"Confusion Matrix: {feature_names[feature_idx]}")
        axes[plot_idx].set_xlabel("Predicted")
        axes[plot_idx].set_ylabel("True")

    for plot_idx in range(len(confusion_matrices), len(axes)):
        fig.delaxes(axes[plot_idx])

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_norm(confusion_matrices, feature_names):
    fig, axes = _init_heatmap_axes(len(confusion_matrices))

    for plot_idx, (feature_idx, payload) in enumerate(confusion_matrices.items()):
        matrix = payload["matrix"]
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100

        sns.heatmap(
            normalized,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=axes[plot_idx],
            cbar_kws={"label": "%"},
            xticklabels=payload["labels"],
            yticklabels=payload["labels"],
            vmin=0,
            vmax=100,
        )
        axes[plot_idx].set_title(f"Confusion Matrix: {feature_names[feature_idx]}\n(Row %)")
        axes[plot_idx].set_xlabel("Predicted")
        axes[plot_idx].set_ylabel("True")

    for plot_idx in range(len(confusion_matrices), len(axes)):
        fig.delaxes(axes[plot_idx])

    plt.tight_layout()
    plt.show()


def inspect_single_prediction(model, dataset, device, active_heads, feature_names, skip_symbols=None):
    skip_symbols = set(skip_symbols or [])
    model.eval()

    with torch.no_grad():
        for embeddings, _, _, ipa_symbols, features in dataset:
            symbol = ipa_symbols[0]
            if symbol in skip_symbols:
                continue

            outputs = model(embeddings.to(device))
            print(f"Symbol: {symbol}")

            for feature_idx in active_heads:
                logits = outputs[feature_idx]
                pred = torch.argmax(logits, dim=-1).item() if logits.dim() == 1 else torch.argmax(logits, dim=1).item()
                print(feature_names[feature_idx], int(features[feature_idx].item()), pred)
            break


def collect_symbol_prediction_stats(model, dataset, device, active_heads, ignore_index=None):
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    true_classes = {}

    model.eval()
    with torch.no_grad():
        for embeddings, _, _, ipa_symbols, features in dataset:
            symbol = ipa_symbols[0]
            outputs = model(embeddings.to(device))

            for feature_idx in active_heads:
                logits = outputs[feature_idx]
                pred = torch.argmax(logits, dim=-1).item() if logits.dim() == 1 else torch.argmax(logits, dim=1).item()
                true_value = int(features[feature_idx].item())
                if ignore_index is not None and true_value == ignore_index:
                    continue
                stats[symbol][feature_idx][pred] += 1
                true_classes[(symbol, feature_idx)] = true_value

    return stats, true_classes


def build_prediction_rows(stats, true_classes):
    rows = []
    for symbol, heads_dict in stats.items():
        for head_id, class_counts in heads_dict.items():
            for pred_class, count in class_counts.items():
                rows.append(
                    {
                        "symbol": symbol,
                        "head": head_id,
                        "class": pred_class,
                        "count": count,
                        "true_class": true_classes[(symbol, head_id)],
                    }
                )
    return rows


def plot_symbol_prediction_distributions(prediction_rows, feature_names):
    df = pd.DataFrame(prediction_rows)

    for symbol, subdf in df.groupby("symbol"):
        true_sub = subdf[["head", "true_class"]].drop_duplicates().sort_values("head")
        head_order = true_sub["head"].astype(int).tolist()
        true_class_by_head = dict(zip(true_sub["head"].astype(int), true_sub["true_class"].astype(int)))
        class_order = sorted(set(subdf["class"].astype(int)) | set(true_class_by_head.values()))

        pivot = (
            subdf.assign(head=subdf["head"].astype(int), pred_class=subdf["class"].astype(int))
            .pivot_table(index="head", columns="pred_class", values="count", aggfunc="sum", fill_value=0)
            .reindex(index=head_order, columns=class_order, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(max(12, len(head_order) * 0.6), 6))
        x = np.arange(len(head_order))
        width = 0.8 / max(1, len(class_order))

        for class_pos, pred_class in enumerate(class_order):
            offsets = x - 0.4 + (class_pos + 0.5) * width
            counts = pivot[pred_class].to_numpy()
            colors = [
                "forestgreen" if true_class_by_head[head_idx] == pred_class else "indianred"
                for head_idx in head_order
            ]
            ax.bar(
                offsets,
                counts,
                width=width,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                label=f"class {pred_class}",
            )

        ax.set_title(f'Prediction distribution for symbol "{symbol}" (green = true class)')
        ax.set_xlabel("Head")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels([feature_names[head] for head in head_order], rotation=90)
        ax.legend(title="Predicted class")
        plt.tight_layout()
        plt.show()

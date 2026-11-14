"""Frame-level feature inference using the thesis checkpoints."""

import numpy as np
import pandas as pd
import torch
from torch import nn


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


EXTRA_FEATURES = "hitone=0, hireg=0, sg=-1, cg=-1, tense=-1, long=-1, velaric=-1"


def parse_features(text):
    values = {}
    for item in text.replace("\n", ",").split(","):
        if item.strip():
            name, value = item.split("=", 1)
            value = int(value)
            if not name.strip() or value not in (-1, 0, 1):
                raise ValueError(f"Ожидается признак со значением −1, 0 или 1: {item}")
            values[name.strip()] = value
    return values


def infer(checkpoint, vectors, table_path, extra=EXTRA_FEATURES, overrides=""):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["model_config"]
    heads = config["active_heads"]
    model = PhonemeFeaturePredictor(config["input_dim"], config["hidden_dim"],
                                    config["num_classes_per_feature"], heads)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = config["input_dim"]
    if vectors.ndim == 2 and vectors.shape[0] == dim and vectors.shape[1] != dim:
        vectors = vectors.T
    if vectors.ndim != 2 or vectors.shape[1] != dim or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError(f"Ожидается непустая матрица эмбеддингов с {dim} признаками на кадр.")
    with torch.inference_mode():
        outputs = model(torch.from_numpy(vectors))
        decode = np.array([-1, 1, 0])
        features = pd.DataFrame({FEATURE_NAMES[i]: decode[outputs[i].argmax(1).numpy()] for i in heads})

    table = pd.read_csv(table_path, index_col=0).replace({"+": 1, "-": -1, "0": 0})
    lookup = features.assign(**{**parse_features(extra), **parse_features(overrides)})
    columns = lookup.columns.intersection(table.columns)
    if table.empty or columns.empty:
        raise ValueError("В таблице IPA нет строк или общих признаков.")
    reference = table[columns].apply(pd.to_numeric).to_numpy()
    rows = []
    for frame, values in enumerate(lookup[columns].to_numpy()):
        distances = np.count_nonzero(reference != values, axis=1)
        distance = distances.min()
        candidates = table.index[distances == distance].tolist()
        symbol = candidates[1] if len(candidates) > 1 and "\u0308" in candidates[0] else candidates[0]
        rows.append((frame, frame * 0.02, symbol, int(distance), len(candidates), ", ".join(candidates)))
    return features, lookup, pd.DataFrame(rows, columns=[
        "Кадр", "Время, с", "IPA", "Расстояние", "Число кандидатов", "Кандидаты"])

"""Evaluate saved baseline weights on the local phonetic-features split."""
import ast
from collections import Counter, OrderedDict
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

ROOT = Path.home() / 'Downloads/Thesis_project_final'
OUTPUT = Path(__file__).resolve().parent
DATA = ROOT / 'PhoneticFeatures_Training/data'


class FloatStorage:
    pass


def rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks):
    assert offset == 0 and shape == (256,) and stride == (1,)
    return storage


class EmbeddingIndex(pickle.Unpickler):
    def find_class(self, module, name):
        return {
            ('torch', 'FloatStorage'): FloatStorage,
            ('torch._utils', '_rebuild_tensor_v2'): rebuild_tensor,
            ('collections', 'OrderedDict'): OrderedDict,
        }[module, name]

    def persistent_load(self, storage):
        kind, dtype, key, device, count = storage
        assert kind == 'storage' and dtype is FloatStorage and device == 'cpu'
        assert count == 256
        return key


def entries(archive, prefix):
    return EmbeddingIndex(io.BytesIO(archive.read(prefix + 'data.pkl'))).load()


def summary(rows):
    return {
        'frames': len(rows),
        'utterance_files': len({row['place'] for row in rows}),
        'triplet_types': dict(Counter('uniform' if len(set(row['label'])) == 1
                                      else 'boundary' for row in rows)),
        'center_labels': dict(Counter(row['label'][1] for row in rows)),
    }


def main():
    app = ast.parse((ROOT / 'app.py').read_text())
    labels = next(ast.literal_eval(node.value) for node in app.body
                  if isinstance(node, ast.Assign)
                  and node.targets[0].id == 'BASELINE_PHONEMES')
    configurations = {
        'VACXUXVEXO': labels,
        'RGCAIQYZHB': labels + ['a1', 'i1', 'u1', 'y1'],
    }
    (OUTPUT / 'class_mapping.json').write_text(
        json.dumps({uid: dict(enumerate(names)) for uid, names in configurations.items()},
                   ensure_ascii=False, indent=2) + '\n')
    split = {}
    places = {}
    paths = [ROOT / 'app.py', ROOT / 'DatasetsModels.py']
    for part in ('train', 'test'):
        path = DATA / f'{part}_embeddings.pt'
        paths.append(path)
        with zipfile.ZipFile(path) as archive:
            prefix = archive.namelist()[0].rsplit('/', 1)[0] + '/'
            rows = entries(archive, prefix)
            split[part] = summary(rows)
            places[part] = {row['place'] for row in rows}
            if part == 'test':
                features = np.vstack([np.frombuffer(
                    archive.read(prefix + 'data/' + row['embedding']), dtype='<f4')
                    for row in rows])
    split['overlapping_utterance_filenames'] = sorted(places['train'] & places['test'])
    scores = {}
    for uid, names in configurations.items():
        path = ROOT / 'models/baseline' / f'{uid}_model.pth'
        paths.append(path)
        with zipfile.ZipFile(path) as archive:
            shapes = [(512, 256), (256, 512), (128, 256), (64, 128)] + [(len(names), 64)] * 3
            layers = [(np.frombuffer(archive.read(f'archive/data/{2*i}'), '<f4').reshape(shape),
                       np.frombuffer(archive.read(f'archive/data/{2*i+1}'), '<f4'))
                      for i, shape in enumerate(shapes)]
        index = {label: i for i, label in enumerate(names)}
        mask = np.array([all(label in index for label in row['label']) for row in rows])
        valid = [row for row, keep in zip(rows, mask) if keep]
        y = np.array([index[row['label'][1]] for row in valid])
        x = features[mask]
        for w, b in layers[:4]:
            x = np.maximum(x @ w.T + b, 0)
        w, b = layers[5]
        pred = (x @ w.T + b).argmax(axis=1)
        cm = np.zeros((len(names), len(names)), dtype=np.int64)
        np.add.at(cm, (y, pred), 1)
        support, predicted, correct = cm.sum(axis=1), cm.sum(axis=0), cm.diagonal()
        observed = support > 0
        union = support + predicted > 0
        scores[uid] = {
            'frames': len(y), 'excluded_frames': int((~mask).sum()),
            'true_classes': int(observed.sum()), 'macro_f1_classes': int(union.sum()),
            'accuracy': float(correct.sum() / len(y)),
            'balanced_accuracy': float((correct[observed] / support[observed]).mean()),
            'macro_f1': float((2 * correct[union] / (support + predicted)[union]).mean()),
        }
        with (OUTPUT / f'{uid}_center_confusion.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['true / predicted', *names])
            writer.writerows([name, *counts] for name, counts in zip(names, cm))
    result = {'numpy': np.__version__, 'split': split, 'local_test_center': scores,
              'sha256': {str(p.relative_to(ROOT)): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
                         for p in paths}}
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(scores, indent=2))
    print({key: {k: v for k, v in value.items() if k != 'center_labels'}
           if isinstance(value, dict) else value for key, value in split.items()})


if __name__ == '__main__':
    main()

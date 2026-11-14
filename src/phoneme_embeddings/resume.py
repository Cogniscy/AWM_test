"""Find compatible completed results without loading an encoder."""
import json
from pathlib import Path

import numpy as np


def signature(config):
    encoder = dict(config['encoder'])
    if 'checkpoint' in encoder:
        encoder['checkpoint'] = str(Path(encoder['checkpoint']).expanduser().resolve())
    return (encoder, config.get('normalization', 'l2'),
            float(config.get('overlap_threshold_sec', 0.001)))


def completed_pairs(model_root, config, on_scan=None):
    """Index all earlier runs, validating files before treating them as complete.

    Legacy runs have no source fingerprints: reject inputs/checkpoints modified
    after run.json. Existing inputs at unchanged paths are required.
    """
    found = set()
    wanted = signature(config)
    for index, run in enumerate(sorted(Path(model_root).rglob('run.json'))):
        if on_scan and index % 250 == 0:
            on_scan(index)
        try:
            meta = json.loads(run.read_text(encoding='utf-8'))
            if signature(meta) != wanted or not set(config['modes']) <= set(meta['modes']):
                continue
            wav, seg = (Path(meta[key]).resolve() for key in ('wav', 'segmentation'))
            sources = [wav, seg]
            checkpoint = meta['encoder'].get('checkpoint')
            if checkpoint:
                sources.append(Path(checkpoint))
            if any(source.stat().st_mtime_ns > run.stat().st_mtime_ns for source in sources):
                continue
            for mode in config['modes']:
                path = run.parent / f'{mode}.npy'
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.ndim != 2 or array.shape[1] != meta['dimension']:
                    raise ValueError('Invalid array shape')
                if path.stat().st_size != array.offset + array.nbytes:
                    raise ValueError('Incomplete array')
                count, rows = 0, 0
                with (run.parent / f'{mode}.jsonl').open(encoding='utf-8') as handle:
                    for line in handle:
                        record = json.loads(line)
                        if record['row'] is not None:
                            if record['row'] != rows:
                                raise ValueError('Invalid row sequence')
                            rows += 1
                        count += 1
                if rows != len(array):
                    raise ValueError('Array/metadata mismatch')
                if mode == 'frame' and (count != rows or rows != meta['frame_count']):
                    raise ValueError('Invalid frame count')
            found.add((wav, seg))
        except (OSError, ValueError, TypeError, KeyError, EOFError, OverflowError):
            # Missing, corrupt and interrupted outputs remain pending.
            continue
    return found

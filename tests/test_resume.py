import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from phoneme_embeddings.resume import completed_pairs


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.wav = self.root / 'speaker_a' / 'same.wav'
        self.wav.parent.mkdir()
        self.wav.write_bytes(b'audio')
        self.seg = self.wav.with_suffix('.seg_B2')
        self.seg.write_text('labels')
        self.config = dict(encoder=dict(name='test'), modes=['frame', 'segment_mean'])
        self.run = self.root / 'results' / 'old_run'
        self.run.mkdir(parents=True)
        for mode in self.config['modes']:
            np.save(self.run / f'{mode}.npy', np.ones((1, 2)))
            (self.run / f'{mode}.jsonl').write_text('{"row":0}\n')
        self.meta = dict(self.config, wav=str(self.wav), segmentation=str(self.seg),
                         dimension=2, frame_count=1)
        (self.run / 'run.json').write_text(json.dumps(self.meta))

    def index(self, config=None):
        return completed_pairs(self.root / 'results', config or self.config)

    def test_resume_and_compatible_subset(self):
        self.assertEqual(self.index(), {(self.wav, self.seg)})
        self.assertEqual(self.index(dict(self.config, modes=['frame'])), {(self.wav, self.seg)})

    def test_changed_settings_and_input(self):
        self.assertFalse(self.index(dict(self.config, normalization='none')))
        self.assertFalse(self.index(dict(self.config, encoder=dict(name='other'))))
        future = (self.run / 'run.json').stat().st_mtime + 10
        os.utime(self.seg, (future, future))
        self.assertFalse(self.index())

    def test_interrupted_and_corrupt_outputs(self):
        (self.run / 'segment_mean.npy').unlink()
        self.assertFalse(self.index())
        np.save(self.run / 'segment_mean.npy', np.ones((1, 2)))
        (self.run / 'frame.jsonl').write_text('{"row":')
        self.assertFalse(self.index())
        (self.run / 'run.json').unlink()
        self.assertFalse(self.index())

    def test_same_basename_is_not_same_record(self):
        other = self.root / 'speaker_b' / 'same.wav'
        self.assertNotIn((other, other.with_suffix('.seg_B2')), self.index())


if __name__ == '__main__':
    unittest.main()

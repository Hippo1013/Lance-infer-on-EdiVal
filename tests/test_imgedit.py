"""Exercise ImgEdit's distinct metadata and two/three-turn session boundaries."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from lance_mice.dataset import IMGEDIT_SPLITS, load_samples, shard_samples
from lance_mice.imgedit import main
from lance_mice.runner import run_session
from lance_mice.settings import Settings
from test_core import FakeBackend


class ImgEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lance-imgedit-')
        self.root = Path(self.tmp.name)
        self.data = self.root / 'multiturn'
        for i, split in enumerate(IMGEDIT_SPLITS):
            folder = self.data / split
            folder.mkdir(parents=True)
            Image.new('RGB', (32, 32), (20 + i, 30, 40)).save(folder / '000038819.jpg')
            row = {'id': '000038819.jpg', 'turn1': 'Keep every subsequent edit yellow.',
                   'turn2': 'Change it.', 'evaluation_answer': 'DO NOT INPUT'}
            if i:
                row['turn3'] = 'Apply this edit to the original image.'
            (folder / 'annotation.json').write_text(json.dumps(row) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_jsonl_leading_zero_category_identity_and_variable_turns(self):
        samples = load_samples(self.root, benchmark='imgedit')
        self.assertEqual([len(s.instructions) for s in samples], [2, 3, 3])
        self.assertEqual(len({s.session_id for s in samples}), 3)
        self.assertTrue(all(s.sample_id == '000038819' for s in samples))
        self.assertNotIn('DO NOT INPUT', str([s.instructions for s in samples]))
        self.assertEqual([len(shard_samples(samples, i, 2)) for i in (0, 1)], [2, 1])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(['--dataset', str(self.root), '--dry-run'])
        spec = json.loads(stream.getvalue())
        self.assertEqual((spec['benchmark'], spec['sessions'], spec['turns']), ('imgedit', 3, 8))
        self.assertEqual(spec['settings']['settings']['cache_mode'], 'prefix')

    def test_two_turn_session_closes_and_resume_never_invents_third_turn(self):
        sample = load_samples(self.data, benchmark='imgedit')[0]
        settings = Settings(cache_mode='prefix')
        backend = FakeBackend(settings)
        flags = []
        edit = backend.edit
        def wrapped(*a, **kw):
            flags.append(kw['end_session'])
            return edit(*a, **kw)
        backend.edit = wrapped
        output = self.root / 'outputs'
        rows = run_session(sample, output, settings, backend, run_id='test')
        self.assertEqual(flags, [False, True])
        self.assertEqual(len(rows), 2)
        self.assertFalse((output / sample.session_id / 'turn_3.png').exists())
        calls = len(backend.calls)
        run_session(sample, output, settings, backend, run_id='test', resume=True)
        self.assertEqual(len(backend.calls), calls)
        (output / sample.session_id / 'turn_3.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unexpected saved turns'):
            run_session(sample, output, settings, backend, run_id='test', resume=True)

    def test_missing_turn_or_path_escape_rejected(self):
        path = self.data / IMGEDIT_SPLITS[0] / 'annotation.json'
        original = json.loads(path.read_text())
        for row in ({**original, 'id': '../000038819.jpg'},
                    {k: v for k, v in original.items() if k != 'turn1'}):
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_samples(self.data, benchmark='imgedit')


if __name__ == '__main__':
    unittest.main()

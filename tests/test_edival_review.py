import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

SPEC = importlib.util.spec_from_file_location('edival_review', Path(__file__).resolve().parents[1] / 'scripts/edival_review.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EdiValReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='umm-edival-review-')
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        samples = [[f'edival/{i}', [f'Instruction {i}/{j}' for j in range(1, 4)]] for i in range(572)]
        (self.run / 'run.json').write_text(json.dumps({'benchmark': 'edival', 'fingerprint': 'run-id',
            'settings': {'settings': {'resolution': 512}}, 'samples': samples}))
        (self.root / 'status.json').write_text(json.dumps({'state': 'running'}))
        self.review = MODULE.Review(self.run, self.root / 'review')

    def tearDown(self):
        self.temp.cleanup()

    def test_pending_and_completed_turn_identity(self):
        manifest = self.review.manifest()
        self.assertEqual(manifest['completed_turns'], 0)
        directory = self.run / 'edival/0'
        directory.mkdir(parents=True)
        instructions = manifest['sessions'][0]['instructions_en']
        (directory / 'session.json').write_text(json.dumps({'session_id': 'edival/0', 'instructions': instructions, 'fingerprint': 'session-id'}))
        for name in ('turn_0_input.png', 'turn_1.png'):
            Image.new('RGB', (512, 512), 'white').save(directory / name)
        meta = {'turn': 1, 'instruction': instructions[0], 'session_fingerprint': 'session-id'}
        (directory / 'turn_1.json').write_text(json.dumps(meta))
        self.assertEqual(self.review.manifest()['completed_turns'], 0)
        (directory / 'turn_1.attention.npz').write_bytes(b'presence; final inference audit validates contents')
        self.assertEqual(self.review.manifest()['completed_turns'], 1)
        meta['instruction'] = 'Different instruction'
        (directory / 'turn_1.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'metadata identity'):
            self.review.manifest()

    def test_translation_provenance_and_publication_gate(self):
        path = self.root / 'translations.json'
        path.write_text(json.dumps({'run_fingerprint': 'other', 'sessions': {}}))
        review = MODULE.Review(self.run, self.root / 'review', path)
        with self.assertRaisesRegex(ValueError, 'different run'):
            review.manifest()
        path.write_text(json.dumps({'run_fingerprint': 'run-id', 'sessions': {'edival/0': {'en': ['wrong'] * 3, 'zh': ['译'] * 3}}}))
        with self.assertRaisesRegex(ValueError, 'sequence differs'):
            review.manifest()
        with self.assertRaisesRegex(ValueError, 'Complete provenance'):
            self.review.finalize()
        self.assertFalse((self.root / 'review').exists())

    def test_image_routes_do_not_escape_run(self):
        for route in ('/image/../../status.json', '/image/edival/0/4', '/image/edival/01/0', '/status.json'):
            self.assertIsNone(self.review.image_path(route))


if __name__ == '__main__':
    unittest.main()

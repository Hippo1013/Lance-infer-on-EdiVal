"""Exercise cumulative CSV rows, archive coverage and actual generated history."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from lance_mice.acceptance import validate_run
from lance_mice.backend import history_payload
from lance_mice.dataset import load_samples, shard_samples
from lance_mice.edival import main
from lance_mice.edival_dataset import CSV_NAME, ZIP_NAME, source_identity
from lance_mice.images import image_hash
from lance_mice.protocol import digest, render_user
from lance_mice.runner import run_session, write_json
from lance_mice.settings import Settings
from test_core import FakeBackend


class EdiValTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lance-edival-')
        self.root = Path(self.tmp.name)
        self.chains = {
            '0': ['Add a bench.', '"Change the count to 4"', 'Change the background.'],
            '1': ['Add fish below the bear.', 'Make the fish glass.', 'Replace it.'],
            '12': ['Add a bird.', 'Replace the bird with a cat.', 'Remove the cat.'],
        }
        self.rows = []
        for identifier, instructions in self.chains.items():
            for turn in (3, 1, 2):
                self.rows.append({'image_index': identifier, 'turns': str(turn),
                    'instructions': repr(instructions[:turn]),
                    'format_instructions': "['SECRET PARSED INSTRUCTION']",
                    'task_type': "['SECRET TASK LABEL']",
                    'all_objects': "['SECRET EVALUATION OBJECT']"})
        self.save_rows()
        with ZipFile(self.root / ZIP_NAME, 'w') as archive:
            for i, identifier in enumerate((*self.chains, '99')):
                stream = io.BytesIO()
                Image.new('RGB', (32, 32), (31 + i, 42, 53)).save(stream, format='JPEG')
                archive.writestr(f'{identifier}_input_raw.jpg', stream.getvalue())

    def tearDown(self):
        self.tmp.cleanup()

    def save_rows(self):
        with (self.root / CSV_NAME).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_prefix_rows_collapse_and_unused_archive_image_is_excluded(self):
        samples = load_samples(self.root, 'all', 'edival')
        self.assertEqual(len(samples), 3)  # Nine CSV rows and four images are three sessions.
        self.assertEqual({s.session_id for s in samples}, {'edival/0', 'edival/1', 'edival/12'})
        for sample in samples:
            self.assertEqual(sample.instructions, tuple(self.chains[sample.sample_id]))
        left, right = [shard_samples(samples, i, 2) for i in range(2)]
        self.assertFalse({s.session_id for s in left} & {s.session_id for s in right})
        self.assertEqual(len(left + right), 3)

    def test_archive_decode_matches_file_decode_and_preserves_source_reference(self):
        sample = load_samples(self.root, 'all', 'edival')[0]
        with ZipFile(sample.image) as archive:
            path = self.root / 'source.jpg'
            path.write_bytes(archive.read(sample.image_member))
        with Image.open(path) as image:
            self.assertEqual(image_hash(sample.read_source()), image_hash(image))
        output = self.root / 'run'
        run_session(sample, output, Settings(), FakeBackend(Settings()), run_id='fixture')
        session = json.loads((output / sample.session_id / 'session.json').read_text())
        self.assertEqual(session['source_member'], sample.image_member)
        self.assertEqual(session['source'], str(sample.image))

    def test_dry_run_defaults_to_dp2_prefix_and_creates_no_outputs(self):
        output = self.root / 'must-not-exist'
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(['--dataset', str(self.root), '--dry-run', '--output', str(output)])
        report = json.loads(stream.getvalue())
        self.assertEqual((report['benchmark'], report['sessions'], report['turns']), ('edival', 2, 6))
        self.assertEqual([len(a) for a in report['assignments']], [1, 1])
        self.assertEqual(report['settings']['settings']['cache_mode'], 'prefix')
        self.assertEqual(report['settings']['settings']['resolution'], 512)
        self.assertFalse(output.exists())

    def test_resolution_is_part_of_run_identity_and_768_defaults_remain(self):
        settings = Settings()
        smaller = Settings(resolution=512)
        self.assertEqual(settings.resolution, 768)
        self.assertNotEqual(settings.fingerprint(), smaller.fingerprint())
        with self.assertRaises(ValueError):
            Settings(resolution=640)

    def test_full_history_contains_only_raw_instructions_and_real_output_pixels(self):
        settings, output = Settings(), self.root / 'run'
        samples = load_samples(self.root, 'smoke', 'edival')
        write_json(output / 'run.json', {'samples': [(s.session_id, s.instructions) for s in samples],
            'settings': settings.identity(), 'model_files': {'fixture': True}})
        backend = FakeBackend(settings)
        for sample in samples:
            rows = run_session(sample, output, settings, backend, run_id='fixture')
            calls = backend.calls[-3:]
            self.assertEqual(calls[1][1][-1], rows[0]['output_hash'])
            self.assertEqual(calls[2][1][-1], rows[1]['output_hash'])
            payload = history_payload(sample.session_id, [sample.read_source()] * 3,
                                      list(sample.instructions), settings)
            self.assertNotIn('SECRET', json.dumps(payload['extra_args']))
            self.assertEqual(render_user(list(sample.instructions)), rows[2]['protocol']['user_prompt_preview'])
            self.assertFalse(any(s['text'] for s in rows[2]['protocol']['segments'] if s['role'] == 'label'))
        self.assertEqual(len(validate_run(output)), 6)

    def test_resume_reconstructs_history_and_rejects_changed_archive_pixels(self):
        sample = load_samples(self.root, 'smoke', 'edival')[0]
        output, settings = self.root / 'resumed', Settings()
        first = run_session(sample, output, settings, FakeBackend(settings), run_id='fixture', max_turns=1)
        backend = FakeBackend(settings)
        run_session(sample, output, settings, backend, run_id='fixture', resume=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0][1][-1], first[0]['output_hash'])
        with ZipFile(sample.image) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        stream = io.BytesIO()
        Image.new('RGB', (32, 32), 'white').save(stream, format='JPEG')
        members[sample.image_member] = stream.getvalue()
        with ZipFile(sample.image, 'w') as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        with self.assertRaisesRegex(ValueError, 'configuration or source changed'):
            run_session(sample, output, settings, FakeBackend(settings), run_id='fixture', resume=True)

    def test_csv_or_zip_changes_invalidate_data_identity(self):
        identity = source_identity(self.root)
        self.rows[0]['all_objects'] = "['CHANGED ANNOTATION']"
        self.save_rows()
        changed = source_identity(self.root)
        self.assertNotEqual(digest(identity), digest(changed))
        self.assertEqual(identity[ZIP_NAME], changed[ZIP_NAME])

    def test_duplicate_missing_and_conflicting_prefixes_are_rejected(self):
        original = list(self.rows)
        self.rows.append(dict(self.rows[0]))
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_samples(self.root, 'all', 'edival')
        self.rows = original[1:]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'Missing prefix'):
            load_samples(self.root, 'all', 'edival')
        self.rows = [dict(r) for r in original]
        self.rows[1]['instructions'] = "['Different first instruction']"
        self.save_rows()
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            load_samples(self.root, 'all', 'edival')

    def test_unsafe_ids_malformed_lists_and_missing_images_are_rejected(self):
        original = [dict(r) for r in self.rows]
        for value in ('../0', '-1', '00'):
            self.rows = [dict(r) for r in original]
            self.rows[0]['image_index'] = value
            self.save_rows()
            with self.assertRaisesRegex(ValueError, 'image_index'):
                load_samples(self.root, 'all', 'edival')
        for value in ('__import__("os").getcwd()', "['Only one']", "['a', '', 'c']"):
            self.rows = [dict(r) for r in original]
            self.rows[0]['instructions'] = value
            self.save_rows()
            with self.assertRaisesRegex(ValueError, 'Invalid instruction'):
                load_samples(self.root, 'all', 'edival')
        self.rows = original
        self.save_rows()
        with ZipFile(self.root / ZIP_NAME, 'w'):
            pass
        with self.assertRaisesRegex(ValueError, 'Missing EdiVal source'):
            load_samples(self.root, 'all', 'edival')


if __name__ == '__main__':
    unittest.main()

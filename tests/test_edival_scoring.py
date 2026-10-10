"""CPU regressions for image routing, official aggregation and swallowed errors."""
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('edival_scoring', Path(__file__).parents[1] / 'scripts/edival_scoring.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class ScoringRoutingTests(unittest.TestCase):
    def test_multipass_uses_previous_for_if_original_for_cc(self):
        class Paths:
            def __init__(self, **kw): self.__dict__.update(kw)
        record = {'images': [{'path': x} for x in ('original', 'edit1', 'edit2', 'edit3')]}
        turn2 = s.paths_for(record, 2, Paths)
        self.assertEqual((turn2.base, turn2.source, turn2.target), ('original', 'edit1', 'edit2'))
        turn3 = s.paths_for(record, 3, Paths)
        self.assertEqual((turn3.base, turn3.source, turn3.target), ('original', 'edit2', 'edit3'))

    def test_mutated_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='edival-score-test.') as tmp:
            p = Path(tmp) / 'image.png'
            p.write_bytes(b'accepted')
            bound = dict(path=str(p), sha256=s.sha(p))
            record = dict(images=[bound], turn_records=[], session_record=bound)
            s.check_record(record)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                s.check_record(record)

    def test_caught_upstream_failure_is_not_a_valid_no(self):
        errors = []
        stream = s.OfficialOutput(io.StringIO(), errors)
        stream.write('VLM Response: no\n')
        self.assertEqual(errors, [])
        stream.write('Error querying VLM: engine ')
        stream.write('failed\n')
        self.assertEqual(errors, ['Error querying VLM: engine failed'])

    def test_nonfinite_nested_score_rejected(self):
        with self.assertRaises(ValueError):
            s.finite({'quality': [1., float('nan')]})

    @unittest.skipUnless(os.environ.get('EDIVAL_OFFICIAL_SOURCE'), 'Requires pinned official source path')
    def test_official_collector_omits_null_cc_and_counts_all_if(self):
        ns = s.official_namespace(Path(os.environ['EDIVAL_OFFICIAL_SOURCE']),
                                 ['LOCAL_TASK_TYPES', 'GLOBAL_TASK_TYPES', 'CONSISTENCY_TASK_TYPES',
                                  'QUALITY_TASK_TYPES', 'TASK_TYPES', 'TaskRateCollector'])
        collector = ns['TaskRateCollector'](3)
        collector.add_score(1, 'subject_add', 0)
        collector.add_score(1, 'subject_add', 1)
        collector.add_score(1, 'object_dinov3_consistency', None)
        collector.add_score(1, 'object_dinov3_consistency', .7)
        self.assertEqual(collector.data['overall'], [0, 1])
        self.assertEqual(collector.data[1]['object_dinov3_consistency'], [.7])

    @unittest.skipUnless(os.environ.get('EDIVAL_OFFICIAL_SOURCE'), 'Requires pinned official source path')
    def test_official_turn_routes_cc_to_base_and_if_to_previous(self):
        calls = []
        def instruction(src, target, **kw):
            calls.append(('if', src, target, kw['instruction']))
            return 0, 'original reason'
        def quality(target, model, **kw):
            calls.append(('quality', target, kw['prompt']))
            return {'human_preference_score': None}
        def consistency(src, target, unchanged, all_objects, **kw):
            calls.append(('cc', src, target, unchanged, all_objects))
            return ({'object_dinov3_consistency_mean': None, 'object_l1_consistency_mean': None},
                    {'bg_l1_consistency': .6, 'bg_dinov3_masked_similarity': .8})
        ns = s.official_namespace(Path(os.environ['EDIVAL_OFFICIAL_SOURCE']),
            ['ImagePaths', 'EvaluationModels', 'evaluate_turn_multipass'],
            dict(evaluate_instruction_following=instruction, evaluate_quality=quality,
                 evaluate_consistency=consistency))
        annotation = dict(instruction='current instruction', format_instruction='current format',
                          task_type='subject_add', unchanged_objects=['tree'], all_objects=['tree', 'bench'],
                          eval_bg_consistency=True)
        row = ns['evaluate_turn_multipass'](ns['EvaluationModels'](1, 2, 3, 4, None),
                  ns['ImagePaths']('original', 'edit1', 'edit2'), annotation, 2, 0)
        self.assertIn(('if', 'edit1', 'edit2', 'current instruction'), calls)
        self.assertIn(('cc', 'original', 'edit2', ['tree'], ['tree', 'bench']), calls)
        self.assertEqual(row['instruction_following_reason'], 'original reason')
        calls.clear()
        annotation['eval_bg_consistency'] = False
        row = ns['evaluate_turn_multipass'](ns['EvaluationModels'](1, 2, 3, 4, None),
                  ns['ImagePaths']('original', 'edit1', 'edit2'), annotation, 2, 0)
        self.assertFalse(any(c[0] == 'cc' for c in calls))
        self.assertIsNone(row['object_details'])


if __name__ == '__main__':
    unittest.main()

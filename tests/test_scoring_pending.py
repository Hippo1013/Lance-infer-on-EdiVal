import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from lance_mice import scoring_pending as pending, scoring_qwen_once as profile
from lance_mice.scoring import core, runner
from lance_mice.images import image_hash
from lance_mice.scoring.request_cache import RequestCache

CONTRADICTORY_GA = '<answer_turn_1> yes </answer_turn_1><reason_turn_1>违反全局约束</reason_turn_1><answer_final> no </answer_final>'


class FakeJudge:
    def __init__(self, if_raw='NO', ga_raw=CONTRADICTORY_GA, failure=None):
        self.if_raw, self.ga_raw, self.failure = if_raw, ga_raw, failure
        self.traces, self.vote = [], 1

    def ask(self, images, prompt, kind):
        if self.failure:
            raise self.failure
        raw = self.ga_raw if kind == 'GA' else self.if_raw
        self.traces.append({'kind': kind, 'raw': raw, 'vote': self.vote,
            'model': '/home/chs/model/Qwen3.6-27B', 'image_hashes': [image_hash(im) for im in images],
            'image_sizes': [list(im.size) for im in images], 'prompt': prompt,
            'request_hash': core.request_identity([image_hash(im) for im in images], prompt, self.vote),
            'sampling_seed': 42, 'temperature': .6, 'runtime_context_limit': 16384,
            'finish_reason': 'stop', 'prompt_tokens': 100, 'completion_tokens': 20})
        return raw


class FakeIncompleteJudge(FakeJudge):
    def ask(self, images, prompt, kind):
        raw = super().ask(images, prompt, kind)
        if kind == 'IF':
            trace = self.traces[-1]
            trace.update(raw='CSS\n' * 1024, finish_reason='length', completion_tokens=2048,
                         error='ValueError: Empty or truncated judge response')
            raise pending.ResponseIncomplete(trace)
        return raw


class PendingReviewTests(unittest.TestCase):
    def setUp(self):
        self.old_votes = core.SETTINGS['votes_per_judge']
        self.old_judges = dict(core.JUDGES)
        core.SETTINGS['votes_per_judge'] = 1
        core.JUDGES.clear()
        core.JUDGES['qwen'] = 'Qwen3.6-27B'
        self.images = [Image.new('RGB', (32, 32), color) for color in ['red', 'green', 'blue', 'black']]
        self.sample = {'session_id': 'cm/example', 'split': 'cm', 'metadata': {
            'instruction': ['Change jacket color'] * 3, 'formatted_instruction': ['Change the color of [jacket] to [black]'] * 3,
            'task_type': ['color_alter'] * 3}}

    def tearDown(self):
        core.SETTINGS['votes_per_judge'] = self.old_votes
        core.JUDGES.clear()
        core.JUDGES.update(self.old_judges)

    def test_original_contradiction_preserves_valid_if_and_next_round(self):
        judge = FakeJudge()
        value = pending.evaluate_turn(self.sample, 2, self.images, {}, None, judge)
        self.assertEqual(value['IF']['score'], 0)
        self.assertEqual(value['GA_prefix']['status'], 'pending_review')
        self.assertIsNone(value['GA_prefix']['score'])
        self.assertEqual([c['kind'] for c in judge.traces], ['IF', 'GA'])
        self.assertEqual(judge.traces[-1]['raw'], CONTRADICTORY_GA)
        judge.ga_raw = ''.join(f'<answer_turn_{t}>yes</answer_turn_{t}>' for t in range(1, 4)) + '<answer_final>yes</answer_final>'
        self.assertEqual(pending.evaluate_turn(self.sample, 3, self.images, {}, None, judge)['GA_prefix']['score'], 1)

    def test_pending_if_still_runs_ga(self):
        judge = FakeJudge(if_raw='YES\nNO', ga_raw='<answer_turn_1>no</answer_turn_1><answer_final>no</answer_final>')
        value = pending.evaluate_turn(self.sample, 1, self.images, {}, None, judge)
        self.assertEqual(value['IF']['status'], 'pending_review')
        self.assertEqual(value['GA_prefix']['score'], 0)
        self.assertEqual([c['kind'] for c in judge.traces], ['IF', 'GA'])

    def test_generation_errors_are_not_manual_decisions(self):
        for error in [RuntimeError('CUDA failure'), ValueError('Empty or truncated judge response')]:
            with self.assertRaises(type(error)):
                pending.evaluate_turn(self.sample, 1, self.images, {}, None, FakeJudge(failure=error))

    def test_first_truncated_answer_is_preserved_and_not_sampled_again(self):
        with tempfile.TemporaryDirectory(prefix='umm-incomplete-test-') as td:
            complete = RequestCache(Path(td) / 'requests', '/home/chs/model/Qwen3.6-27B', {'revision': 'r'}, 'producer')
            judge = pending.PreservedJudge('/home/chs/model/Qwen3.6-27B', complete, Path(td) / 'attempts', 16384)
            calls = []
            def backend(instance, images, prompt, kind):
                calls.append(kind)
                fake = FakeIncompleteJudge(ga_raw='<answer_turn_1>no</answer_turn_1><answer_final>no</answer_final>')
                try:
                    return fake.ask(images, prompt, kind)
                except pending.ResponseIncomplete:
                    raise ValueError('Empty or truncated judge response')
                finally:
                    instance.traces.extend(fake.traces)
            with patch.object(pending.Judge, 'ask', backend):
                value = pending.evaluate_turn(self.sample, 1, self.images, {}, None, judge)
                self.assertEqual(value['IF']['status'], 'response_incomplete')
                self.assertIsNone(value['IF']['score'])
                self.assertEqual(value['GA_prefix']['score'], 0)
                self.assertEqual(calls, ['IF', 'GA'])
                first = dict(judge.traces[0])
                with self.assertRaises(pending.ResponseIncomplete):
                    judge.ask(self.images[:2], first['prompt'], 'IF')
                self.assertEqual(calls, ['IF', 'GA'])
                self.assertEqual(judge.traces[-1]['raw'], first['raw'])
                self.assertIn('reused_from', judge.traces[-1])
                self.assertIsNone(complete.get(first['request_hash']))
                self.assertEqual(judge.incomplete_cache.get(first['request_hash'])['trace']['completion_tokens'], 2048)
                with self.assertRaises(ValueError):
                    judge.incomplete_cache.put({**first, 'raw': 'different'})

    def test_pending_cumulative_scores_and_valid_denominator(self):
        values = pending.cumulative([core.result('ok', 0), pending.pending('ambiguous'), core.result('ok', 1)])
        self.assertEqual([v['score'] for v in values], [0, None, None])
        self.assertEqual(values[2]['unresolved_prefixes'], [2])
        self.assertEqual(core.mean_report(values)['valid'], 1)
        self.assertEqual([v['score'] for v in pending.cumulative([core.result('ok', 1), core.result('ok', 0), core.result('ok', 1)])], [1, 0, 0])

    def test_one_vote_and_no_format_repair(self):
        self.assertEqual(pending.ga_result(CONTRADICTORY_GA, 2)['reason'], 'GA must stop at its first failure')
        self.assertEqual(pending.ga_result('<answer_final>no</answer_final>', 2)['status'], 'pending_review')
        with self.assertRaises(ValueError):
            pending.vote_mean([core.result('ok', 1), core.result('ok', 0)])

    def test_report_and_audit_include_pending_with_complete_coverage(self):
        with tempfile.TemporaryDirectory(prefix='umm-pending-test-') as td:
            output = Path(td)
            sample = dict(self.sample)
            sample['images'] = []
            sample['pixel_hashes'] = [image_hash(im) for im in self.images]
            for turn, image in enumerate(self.images):
                path = output / f'image{turn}.png'
                image.save(path)
                sample['images'].append(str(path))
            model_pin = {'name': 'Qwen3.6-27B', 'revision': 'fixture'}
            manifest = {'fingerprint': 'fixture', 'selection': 'all', 'records': [sample],
                        'models': {'models': [model_pin]}}
            for turn in range(1, 4):
                raw = CONTRADICTORY_GA if turn == 2 else ''.join(f'<answer_turn_{i}>yes</answer_turn_{i}>' for i in range(1, turn + 1)) + '<answer_final>yes</answer_final>'
                judge = (FakeIncompleteJudge if turn == 3 else FakeJudge)(ga_raw=raw)
                shared = {'fingerprint': 'fixture', 'status': 'ok', 'session_id': sample['session_id'],
                          'turn': turn, 'CC': core.result('ok', .8), 'detection_requests': {}}
                value = pending.evaluate_turn(sample, turn, self.images, shared, None, judge)
                for trace in judge.traces:
                    if pending.incomplete_trace(trace):
                        pending.IncompleteResponseCache(output / 'response_attempts/qwen', trace['model'],
                            model_pin, 'fixture').put(trace)
                judgment = {'fingerprint': 'fixture', 'status': 'ok', 'session_id': sample['session_id'],
                            'turn': turn, 'judge': 'qwen', 'traces': judge.traces, **value}
                runner.persist(runner.artifact_path(output, 'metrics', sample, turn), shared)
                runner.persist(runner.artifact_path(output, 'qwen', sample, turn), judgment)
            profile.report(SimpleNamespace(output=output), manifest)
            profile.audit(output, manifest)
            summary = core.read(output / 'summary.json')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['pending_components'], 2)
            self.assertEqual(summary['decision_pending_components'], 1)
            self.assertEqual(summary['incomplete_response_components'], 1)
            self.assertEqual(summary['pending_cumulative_ga_turns'], 2)
            self.assertEqual(summary['groups']['all/turn_all']['GA_qwen']['valid'], 1)
            self.assertEqual(core.read(output / 'pending_review.json')['items'][0]['traces'][0]['raw'], CONTRADICTORY_GA)
            index = core.read(output / 'pending_review.json')
            index['items'] = []
            core.write(output / 'pending_review.json', index)
            with self.assertRaisesRegex(ValueError, 'Manual review index'):
                profile.audit(output, manifest)


if __name__ == '__main__':
    unittest.main()

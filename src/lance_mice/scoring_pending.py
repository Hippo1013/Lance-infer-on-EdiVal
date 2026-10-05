"""Completed, ambiguous judge decisions await humans without stopping the pass.

Runtime, dependency and integrity failures still propagate. The original
strict parsers remain the authority; no answer is repaired or sampled again.
"""
from __future__ import annotations

import json
import time

from .scoring import core, runner
from .scoring.judge import Judge
from .scoring.request_cache import RequestCache, valid_request
from .images import image_hash

POLICY = 'completed-decision-pending-review-v1'
INCOMPLETE_POLICY = 'judge-response-incomplete-v1'
MANUAL_STATUSES = {'pending_review', 'response_incomplete'}


def incomplete_trace(trace):
    return (trace.get('error') == 'ValueError: Empty or truncated judge response'
            and isinstance(trace.get('raw'), str)
            and isinstance(trace.get('prompt_tokens'), int)
            and isinstance(trace.get('completion_tokens'), int)
            and (trace.get('finish_reason') == 'length' or
                 (trace.get('finish_reason') == 'stop' and not trace['raw'].strip())))


def incomplete_result(trace):
    if not incomplete_trace(trace):
        raise ValueError('Not a recorded incomplete model response')
    return core.result('response_incomplete', reason=trace['error'],
                       cause='token_limit' if trace['finish_reason'] == 'length' else 'empty_response',
                       finish_reason=trace['finish_reason'], completion_tokens=trace['completion_tokens'],
                       policy=INCOMPLETE_POLICY)


class ResponseIncomplete(ValueError):
    def __init__(self, trace):
        super().__init__(trace['error'])
        self.trace = trace


class IncompleteResponseCache(RequestCache):
    """Separate immutable evidence; never masquerades as a complete answer."""
    def get(self, request_hash):
        path = self.root / (core.digest(self.binding(request_hash)) + '.json')
        if not path.exists():
            return None
        row = core.read(path)
        checksum = row.pop('checksum')
        if (core.digest(row) != checksum or row['binding'] != self.binding(request_hash) or
                not valid_request(row['trace'], self.model) or not incomplete_trace(row['trace'])):
            raise ValueError('Incomplete response evidence identity mismatch')
        return row

    def put(self, trace, provenance=None):
        if not incomplete_trace(trace) or not valid_request(trace, self.model):
            raise ValueError('Only an identified incomplete response can be preserved here')
        existing = self.get(trace['request_hash'])
        if existing:
            if any(existing['trace'][k] != trace[k] for k in ('raw', 'finish_reason', 'completion_tokens')):
                raise ValueError('Conflicting incomplete responses; refusing to select an attempt')
            return
        binding = self.binding(trace['request_hash'])
        row = {'binding': binding, 'trace': trace, 'producer_fingerprint': self.producer,
               'provenance': provenance or {'kind': 'first incomplete model response'}}
        core.write(self.root / (core.digest(binding) + '.json'), {**row, 'checksum': core.digest(row)})


class PreservedJudge(Judge):
    def __init__(self, model, cache, incomplete_root, context_limit=None):
        super().__init__(model, cache, context_limit)
        self.incomplete_cache = IncompleteResponseCache(incomplete_root, model, cache.pin, cache.producer)

    def ask(self, images, prompt, kind):
        key = core.request_identity([image_hash(im.convert('RGB')) for im in images], prompt, self.vote)
        existing = self.incomplete_cache.get(key)
        if existing:
            if self.cache.get(key):
                raise ValueError('Both complete and incomplete answers exist for a single vote')
            trace = {**existing['trace'], 'reused_from': {
                'producer_fingerprint': existing['producer_fingerprint'], 'provenance': existing['provenance']}}
            self.traces.append(trace)
            raise ResponseIncomplete(trace)
        try:
            return super().ask(images, prompt, kind)
        except ValueError:
            if self.traces and incomplete_trace(self.traces[-1]):
                trace = self.traces[-1]
                self.incomplete_cache.put(trace)
                raise ResponseIncomplete(trace)
            raise


class DecisionPending(ValueError):
    """Only an interpretation failure of an already completed response."""


def explicit_decision(raw):
    try:
        return core.parse_yes_no(raw)
    except ValueError as exc:
        raise DecisionPending(str(exc)) from exc


def pending(reason):
    return core.result('pending_review', reason=reason, policy=POLICY)


def ga_result(raw, turn):
    try:
        score = core.parse_ga(raw, turn)
    except ValueError as exc:
        return pending(str(exc))
    return core.result('ok', score)


def vote_mean(rows):
    if len(rows) != core.SETTINGS['votes_per_judge']:
        raise ValueError('Every configured vote is required')
    if any(row['status'] == 'response_incomplete' for row in rows):
        return core.result('response_incomplete', reason='A vote has an incomplete response',
                           policy=INCOMPLETE_POLICY, components=[r['status'] for r in rows])
    if any(row['status'] == 'pending_review' for row in rows):
        return pending('A vote requires manual review') | {'components': [r['status'] for r in rows]}
    return core.vote_mean(rows)


def cumulative(rows):
    out, unresolved = [], []
    value = 1.0
    for turn, row in enumerate(rows, 1):
        if row['status'] != 'ok':
            unresolved.append(turn)
        if unresolved:
            statuses = {rows[t - 1]['status'] for t in unresolved}
            status = ('response_incomplete' if 'response_incomplete' in statuses else
                      'pending_review' if 'pending_review' in statuses else 'unavailable')
            out.append(core.result(status, reason='A prefix requires resolution',
                                   unresolved_prefixes=list(unresolved)))
        else:
            value = min(value, row['score'])
            out.append(core.result('ok', value))
    return out


def if_result(sample, turn, images, detector, judge):
    def one(image, prompt):
        raw = judge.ask([image], prompt, 'IF')
        return raw.strip().lower() if 'Output only the text content' in prompt else explicit_decision(raw)

    def two(first, second, prompt):
        # ask() is deliberately outside explicit_decision's exception handler.
        return explicit_decision(judge.ask([first, second], prompt, 'IF'))

    try:
        return runner.infer_if(sample, turn, images, detector, one, two)
    except DecisionPending as exc:
        return pending(str(exc))
    except ResponseIncomplete as exc:
        return incomplete_result(exc.trace)


def evaluate_turn(sample, turn, images, shared, detector, judge):
    votes = []
    prompt = core.ga_prompt(sample['split'], sample['metadata']['instruction'][:turn],
                            sample['metadata']['formatted_instruction'][:turn])
    for vote in range(1, core.SETTINGS['votes_per_judge'] + 1):
        judge.vote = vote
        item = {'vote': vote, 'IF': shared.get('shared_if')}
        if item['IF'] is None:
            item['IF'] = if_result(sample, turn, images, detector, judge)
        # A pending IF does not suppress GA or any later round.
        try:
            raw = judge.ask(images[:turn + 1], prompt, 'GA')
            item['GA_prefix'] = ga_result(raw, turn)
        except ResponseIncomplete as exc:
            item['GA_prefix'] = incomplete_result(exc.trace)
        votes.append(item)
    return {'votes': votes,
            'IF': shared['shared_if'] if 'shared_if' in shared else vote_mean([v['IF'] for v in votes]),
            'GA_prefix': vote_mean([v['GA_prefix'] for v in votes])}


def pending_entries(judgment):
    entries = []
    for vote in judgment['votes']:
        for metric in ('IF', 'GA_prefix'):
            value = vote[metric]
            if value['status'] in MANUAL_STATUSES:
                kind = 'GA' if metric == 'GA_prefix' else 'IF'
                traces = [c for c in judgment['traces'] if c['kind'] == kind and c['vote'] == vote['vote']]
                entries.append({'session_id': judgment['session_id'], 'turn': judgment['turn'],
                                'judge': judgment['judge'], 'vote': vote['vote'], 'metric': metric,
                                'status': value['status'], 'score': None, 'reason': value['reason'],
                                'cause': value.get('cause', 'decision_format'),
                                'traces': traces, 'human_score': None, 'human_notes': ''})
    return entries


def judge_stage(args, manifest):
    from .scoring.metrics import Detector
    from .scoring.request_cache import RequestCache
    name = args.judge
    if not name:
        raise ValueError('--judge required')
    for sample in manifest['records']:
        for turn in range(1, 4):
            value = runner.saved(runner.artifact_path(args.output, 'metrics', sample, turn), manifest['fingerprint'])
            if not value or value['status'] != 'ok':
                raise ValueError('Missing metrics stage')
    detector = Detector(args.model_root, args.output / 'detections', manifest['fingerprint'], readonly=True)
    judge, completed, pending_count, incomplete_count = None, 0, 0, 0
    for sample in manifest['records']:
        images = None
        for turn in range(1, 4):
            path = runner.artifact_path(args.output, name, sample, turn)
            out = runner.saved(path, manifest['fingerprint'], args.retry_errors)
            if not out:
                if images is None:
                    images = [runner.open_image(p) for p in sample['images']]
                if judge is None:
                    pin = next(x for x in manifest['models']['models'] if x['name'] == core.JUDGES[name])
                    cache = RequestCache(args.output / 'requests' / name, args.model_root / core.JUDGES[name], pin,
                                         manifest['fingerprint'])
                    judge = PreservedJudge(args.model_root / core.JUDGES[name], cache,
                                           args.output / 'response_attempts' / name,
                                           context_limit=manifest['context_limits'][name])
                judge.traces = []
                out = {'fingerprint': manifest['fingerprint'], 'status': 'ok', 'session_id': sample['session_id'],
                       'turn': turn, 'judge': name, 'decision_policy': POLICY}
                try:
                    shared = runner.saved(runner.artifact_path(args.output, 'metrics', sample, turn), manifest['fingerprint'])
                    out.update(evaluate_turn(sample, turn, images, shared, detector, judge))
                except Exception as exc:
                    out.update(status='error', error=f'{type(exc).__name__}: {exc}', traces=judge.traces)
                    runner.persist(path, out)
                    raise
                out['traces'] = judge.traces
                runner.persist(path, out)
                entries = pending_entries(out)
                if entries:
                    with (args.output / 'pending_review_events.jsonl').open('a') as stream:
                        for entry in entries:
                            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                print(name, sample['session_id'], turn, 'pending_review' if entries else 'ok', flush=True)
            completed += 1
            manual = pending_entries(out)
            pending_count += len(manual)
            incomplete_count += sum(v['status'] == 'response_incomplete' for v in manual)
            core.write(args.output / 'judge_progress.json', {
                'status': 'running', 'completed_turns': completed, 'total_turns': 3 * len(manifest['records']),
                'pending_components': pending_count, 'last_session': sample['session_id'],
                'incomplete_response_components': incomplete_count,
                'last_turn': turn, 'updated_at': time.time()})
    core.write(args.output / 'judge_progress.json', {'status': 'complete', 'completed_turns': completed,
        'total_turns': completed, 'pending_components': pending_count,
        'incomplete_response_components': incomplete_count, 'updated_at': time.time()})

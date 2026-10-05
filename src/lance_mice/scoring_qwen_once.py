"""Independent Qwen-only, one-vote profile over the unchanged scoring primitives."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from .scoring import core, runner
from . import scoring_pending as pending

PROTOCOL = 'mice-qwen-once-v3'
BASE_SOURCE_PINS = core.source_pins
PROFILE_FILES = ('src/lance_mice/scoring_qwen_once.py', 'scripts/score_mice_qwen_once.py',
                 'scripts/run_mice_qwen_scoring.sh', 'scripts/full_mice_qwen_once_job.py',
                 'scripts/full_mice_scoring_job.py', 'src/lance_mice/scoring_pending.py')
V1_PROFILE_PINS = {
    'src/lance_mice/scoring_qwen_once.py': '806dbee686d49de9fc995232d38e55ad3ff0901fb5ac73651c0ba466f9ad356e',
    'scripts/full_mice_qwen_once_job.py': '33e37e1f064a9468dd410bce1680d155816f6400fcf4a3c1d8ebaac96249cb30',
}
V2_PROFILE_PINS = {
    'src/lance_mice/scoring_qwen_once.py': '7c9a89ce5cbcad244e996160375a6dc28863bd97aeddf47a663e32515621af21',
    'scripts/full_mice_qwen_once_job.py': '188273865f2967867dbc1fd14bcbc986eb4fc77f0683681025d5d29286bc02c5',
    'src/lance_mice/scoring_pending.py': '10b0b1940b3dbe648b4f9e1282b9ebcc6b3ed59521cd6d54352a722521866aa4',
}


def source_pins():
    return {**BASE_SOURCE_PINS(), **{name: core.sha(core.ROOT / name) for name in PROFILE_FILES}}


def configure():
    # This changes only this dedicated process; the dual-judge source is untouched.
    core.SETTINGS['votes_per_judge'] = 1
    core.JUDGES.clear()
    core.JUDGES['qwen'] = 'Qwen3.6-27B'
    core.PROTOCOL = runner.PROTOCOL = PROTOCOL
    core.source_pins = runner.source_pins = source_pins


def import_metrics(source, output):
    """Rebind verified, unchanged measurements; preserve every producer hash."""
    source = source.resolve()
    with (source / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = core.read(source / 'manifest.json')
        old_fp = old.pop('fingerprint')
        new = core.read(output / 'manifest.json')
        if core.digest(old) != old_fp or old['protocol'] not in ('mice-dual-judge-v2', 'mice-qwen-once-v1', 'mice-qwen-once-v2'):
            raise ValueError('Unsupported or changed producer manifest')
        previous_qwen = old['protocol'] in ('mice-qwen-once-v1', 'mice-qwen-once-v2')
        legacy_pins = V2_PROFILE_PINS if old['protocol'] == 'mice-qwen-once-v2' else V1_PROFILE_PINS
        if not (source / 'completion.json').exists():
            raise ValueError('Producer must have stopped before measurement reuse')
        completion = core.read(source / 'completion.json')
        if previous_qwen:
            if completion['state'] != 'failed' or completion['fingerprint'] != old_fp:
                raise ValueError('Expected the failed, pinned Qwen producer')
        elif not (source / 'superseded.json').exists():
            raise ValueError('Dual producer was not explicitly stopped')
        if old['records'] != new['records'] or old['dataset_hashes'] != new['dataset_hashes']:
            raise ValueError('Metric inputs changed')
        if old['models'] != new['models']:
            raise ValueError('Model revisions changed')
        for name, pin in old['sources'].items():
            if previous_qwen and name in legacy_pins:
                if pin != legacy_pins[name]:
                    raise ValueError('Unsupported legacy Qwen implementation: ' + name)
            elif new['sources'].get(name) != pin:
                raise ValueError('The original scoring primitives changed: ' + name)
        settings = dict(old['settings'])
        if settings['votes_per_judge'] != (1 if previous_qwen else 2):
            raise ValueError('Unexpected producer sampling profile')
        settings['votes_per_judge'] = 1
        if settings != new['settings']:
            raise ValueError('Changes beyond the authorized vote count')
        if (core.read(source / 'preflight.json')['inference_validation_sha256'] !=
                core.read(output / 'preflight.json')['inference_validation_sha256']):
            raise ValueError('Inference audit changed')
        detections = 0
        for path in sorted((source / 'detections').glob('*.json')):
            row = core.read(path)
            if row['fingerprint'] != old_fp or core.digest(row['detections']) != row['checksum']:
                raise ValueError('Corrupt detection evidence: ' + str(path))
            request = row['request']
            key = core.digest([request[k] for k in ('image_hash', 'target', 'threshold',
                                                  'return_all', 'delete_large_box')])
            if key != path.stem:
                raise ValueError('Detection request identity changed')
            row.update(fingerprint=new['fingerprint'], reused_from={
                'path': str(path), 'sha256': core.sha(path), 'fingerprint': old_fp})
            core.write(output / 'detections' / path.name, row)
            detections += 1
        metrics = 0
        for sample in new['records']:
            for turn in range(1, 4):
                path = runner.artifact_path(source, 'metrics', sample, turn)
                value = runner.saved(path, old_fp)
                if value is None:
                    continue
                if value['status'] != 'ok' or value['session_id'] != sample['session_id'] or value['turn'] != turn:
                    raise ValueError('Invalid shared measurement')
                value.pop('checksum')
                for key, request in value['detection_requests'].items():
                    detection = core.read(output / 'detections' / f'{key}.json')
                    if detection['request'] != request:
                        raise ValueError('Measurement detection dependency changed')
                value.update(fingerprint=new['fingerprint'], reused_from={
                    'path': str(path), 'sha256': core.sha(path), 'fingerprint': old_fp})
                runner.persist(runner.artifact_path(output, 'metrics', sample, turn), value)
                metrics += 1
        answers, judgments, incomplete_answers = 0, 0, 0
        if previous_qwen:
            from .scoring.request_cache import RequestCache
            pin = next(x for x in new['models']['models'] if x['name'] == 'Qwen3.6-27B')
            model = Path('/home/chs/model/Qwen3.6-27B')
            old_cache = RequestCache(source / 'requests/qwen', model, pin, old_fp)
            new_cache = RequestCache(output / 'requests/qwen', model, pin, old_fp)
            incomplete_cache = pending.IncompleteResponseCache(output / 'response_attempts/qwen', model, pin, old_fp)
            for path in sorted((source / 'requests/qwen').glob('*.json')):
                entry = core.read(path)
                checked = old_cache.get(entry['trace']['request_hash'])
                if not checked or path.stem != core.digest(checked['binding']):
                    raise ValueError('Corrupt source request cache')
                new_cache.put(checked['trace'], {'kind': 'immutable Qwen response reuse',
                    'path': str(path), 'sha256': core.sha(path), 'producer_provenance': checked['provenance']})
                answers += 1
            for sample in new['records']:
                for turn in range(1, 4):
                    path = runner.artifact_path(source, 'qwen', sample, turn)
                    if not path.exists():
                        continue
                    entry = core.read(path)
                    checksum = entry.pop('checksum')
                    if core.digest(entry) != checksum or entry['fingerprint'] != old_fp:
                        raise ValueError('Corrupt previous Qwen result')
                    for trace in entry.get('traces', []):
                        if pending.incomplete_trace(trace):
                            if new_cache.get(trace['request_hash']):
                                raise ValueError('Conflicting complete/incomplete producer answers')
                            incomplete_cache.put(trace, {'kind': 'preserved first incomplete response',
                                'path': str(path), 'sha256': core.sha(path), 'fingerprint': old_fp})
                            incomplete_answers += 1
                            continue
                        cached = new_cache.get(trace['request_hash'])
                        if not cached or cached['trace']['raw'] != trace['raw']:
                            raise ValueError('Previous Qwen answer was not preserved')
                    if entry['status'] == 'error':
                        # Interpret its complete cache in the new judge stage; never regenerate it.
                        if entry['error'] not in ('ValueError: GA must stop at its first failure',
                                                   'ValueError: Empty or truncated judge response'):
                            raise ValueError('Unsupported technical producer error')
                        if entry['error'].endswith('Empty or truncated judge response') and not any(
                                pending.incomplete_trace(t) for t in entry['traces']):
                            raise ValueError('Missing first incomplete response evidence')
                        core.write(output / 'previous_error.json', {'path': str(path),
                            'sha256': core.sha(path), 'fingerprint': old_fp, 'evidence': entry})
                        continue
                    if entry['status'] != 'ok' or [v['vote'] for v in entry['votes']] != [1]:
                        raise ValueError('Invalid previous Qwen vote')
                    ga_calls = [c for c in entry['traces'] if c['kind'] == 'GA']
                    if (len(ga_calls) != 1 or pending.ga_result(ga_calls[0]['raw'], turn) != entry['votes'][0]['GA_prefix'] or
                            pending.vote_mean([entry['votes'][0]['GA_prefix']]) != entry['GA_prefix']):
                        raise ValueError('Previous Qwen GA interpretation changed')
                    entry.update(fingerprint=new['fingerprint'], reused_from={
                        'path': str(path), 'sha256': core.sha(path), 'fingerprint': old_fp})
                    runner.persist(runner.artifact_path(output, 'qwen', sample, turn), entry)
                    judgments += 1
        value = {'status': 'passed', 'source': str(source), 'source_fingerprint': old_fp,
                 'source_manifest_sha256': core.sha(source / 'manifest.json'),
                 'fingerprint': new['fingerprint'], 'metrics_reused': metrics,
                 'detections_reused': detections, 'judge_answers_reused': answers,
                 'judge_turns_reused': judgments, 'decision_policy': pending.POLICY,
                 'incomplete_answers_preserved': incomplete_answers,
                 'checks': ['stopped producer and exclusive lock', 'identical original scoring source pins',
                            'identical images, metadata and model revisions', 'every reused result checksum',
                            'detection identities and dependencies', 'per-record producer file hash']}
        core.write(output / 'metric_reuse.json', value)
        if metrics == 3 * len(new['records']):
            core.write(output / 'status_metrics.json', {'status': 'complete',
                'method': 'verified measurement reuse', 'source': str(source)})
        print(json.dumps(value), flush=True)


def report(args, manifest):
    rows, nonmonotonic, review_entries = [], [], []
    for sample in manifest['records']:
        judgments = []
        for turn in range(1, 4):
            value = runner.saved(runner.artifact_path(args.output, 'qwen', sample, turn), manifest['fingerprint'])
            if value is None or value['status'] != 'ok':
                raise ValueError('Missing Qwen result')
            judgments.append(value)
        cumulative = pending.cumulative([r['GA_prefix'] for r in judgments])
        for turn, judgment in enumerate(judgments, 1):
            review_entries.extend(pending.pending_entries(judgment))
            shared = runner.saved(runner.artifact_path(args.output, 'metrics', sample, turn), manifest['fingerprint'])
            if shared is None or shared['status'] != 'ok':
                raise ValueError('Missing shared metric')
            row = {'session_id': sample['session_id'], 'split': sample['split'], 'turn': turn,
                   'task_type': sample['metadata']['task_type'][turn - 1],
                   'instruction': sample['metadata']['instruction'][turn - 1],
                   'IF_qwen': judgment['IF'], 'CC': shared['CC'],
                   'GA_qwen': cumulative[turn - 1], 'GA_prefix_qwen': judgment['GA_prefix'],
                   'IF_votes_qwen': [v['IF'] for v in judgment['votes']],
                   'GA_prefix_votes_qwen': [v['GA_prefix'] for v in judgment['votes']]}
            if (cumulative[turn - 1]['status'] == judgment['GA_prefix']['status'] == 'ok' and
                    cumulative[turn - 1]['score'] < judgment['GA_prefix']['score']):
                nonmonotonic.append([sample['session_id'], turn, 'qwen'])
            rows.append(row)
    groups = {}
    for split in ('all', 'cm', 'cu'):
        for turn in ('all', 1, 2, 3):
            subset = [r for r in rows if (split == 'all' or r['split'] == split) and
                      (turn == 'all' or r['turn'] == turn)]
            groups[f'{split}/turn_{turn}'] = {key: core.mean_report([r[key] for r in subset])
                for key in ('IF_qwen', 'CC', 'GA_qwen', 'GA_prefix_qwen')}
    summary = {'status': 'complete', 'protocol': PROTOCOL, 'scope': manifest['selection'],
               'fingerprint': manifest['fingerprint'], 'judges': ['qwen'], 'votes_per_judge': 1,
               'sessions': len(manifest['records']), 'turns': len(rows), 'groups': groups,
               'nonmonotonic_prefix_judgments': nonmonotonic, 'errors': [],
               'decision_policy': pending.POLICY, 'pending_components': len(review_entries),
               'incomplete_response_components': sum(v['status'] == 'response_incomplete' for v in review_entries),
               'decision_pending_components': sum(v['status'] == 'pending_review' for v in review_entries),
               'incomplete_response_policy': pending.INCOMPLETE_POLICY,
               'manual_review_status': 'pending' if review_entries else 'none',
               'pending_cumulative_ga_turns': sum(r['GA_qwen']['status'] in pending.MANUAL_STATUSES for r in rows),
               'pending_review_index': 'pending_review.json',
               'human_calibration': 'pending',
               'claim': 'Project Qwen single-vote scores; not official-paper or dual-judge scores.'}
    core.write(args.output / 'scores.json', rows)
    core.write(args.output / 'summary.json', summary)
    core.write(args.output / 'pending_review.json', {'status': 'pending' if review_entries else 'none',
        'policy': pending.POLICY, 'count': len(review_entries), 'items': review_entries,
        'cumulative_dependencies': [{'session_id': r['session_id'], 'turn': r['turn'],
            'unresolved_prefixes': r['GA_qwen']['unresolved_prefixes']} for r in rows
            if r['GA_qwen']['status'] in pending.MANUAL_STATUSES]})
    runner.build_review(args.output, manifest, rows)
    review = args.output / 'review.html'
    review.write_text(review.read_text().replace('MICE 小样本裁判核对', 'MICE Qwen 单次全量评分')
        .replace('两位裁判', 'Qwen 裁判').replace('null 为缺项', 'pending_review 等待人工复核；null 为缺项或待定'))
    print(json.dumps({k: summary[k] for k in ('status', 'protocol', 'sessions', 'turns')}), flush=True)


def audit(output, manifest):
    summary = core.read(output / 'summary.json')
    scores = core.read(output / 'scores.json')
    if (summary['status'] != 'complete' or summary['protocol'] != PROTOCOL or
            summary['judges'] != ['qwen'] or summary['votes_per_judge'] != 1 or
            len(scores) != 3 * len(manifest['records'])):
        raise ValueError('Incomplete single-judge report')
    indexed = {(r['session_id'], r['turn']): r for r in scores}
    if len(indexed) != len(scores):
        raise ValueError('Duplicate reported turn')
    traces, expected_pending = 0, []
    for sample in manifest['records']:
        prefixes = []
        for turn in range(1, 4):
            metric = runner.saved(runner.artifact_path(output, 'metrics', sample, turn), manifest['fingerprint'])
            judgment = runner.saved(runner.artifact_path(output, 'qwen', sample, turn), manifest['fingerprint'])
            if not metric or not judgment or metric['status'] != 'ok' or judgment['status'] != 'ok':
                raise ValueError('Incomplete actual measurement')
            for key, request in metric['detection_requests'].items():
                detection = core.read(output / 'detections' / f'{key}.json')
                if (detection['fingerprint'] != manifest['fingerprint'] or detection['request'] != request or
                        core.digest(detection['detections']) != detection['checksum']):
                    raise ValueError('Detection integrity failure')
            if [v['vote'] for v in judgment['votes']] != [1]:
                raise ValueError('Exactly one vote is required')
            ga_calls = [c for c in judgment['traces'] if c['kind'] == 'GA']
            if len(ga_calls) != 1:
                raise ValueError('Exactly one GA call is required')
            call = ga_calls[0]
            if (call['image_hashes'] != sample['pixel_hashes'][:turn + 1] or
                    call['prompt'] != core.ga_prompt(sample['split'], sample['metadata']['instruction'][:turn],
                                                    sample['metadata']['formatted_instruction'][:turn])):
                raise ValueError('GA full-prefix input mismatch')
            vote = judgment['votes'][0]
            ga_value = pending.incomplete_result(call) if pending.incomplete_trace(call) else pending.ga_result(call['raw'], turn)
            if ga_value != vote['GA_prefix']:
                raise ValueError('Raw GA parsing mismatch')
            if judgment['GA_prefix'] != pending.vote_mean([vote['GA_prefix']]):
                raise ValueError('Single-vote GA mismatch')
            if_calls = [c for c in judgment['traces'] if c['kind'] == 'IF']
            if 'shared_if' in metric:
                if if_calls or judgment['IF'] != metric['shared_if'] or vote['IF'] != metric['shared_if']:
                    raise ValueError('Shared IF mismatch')
            else:
                if [c['vote'] for c in if_calls] not in ([], [1]) or judgment['IF'] != pending.vote_mean([vote['IF']]):
                    raise ValueError('Single-vote IF mismatch')
                if vote['IF']['status'] == 'pending_review':
                    if len(if_calls) != 1:
                        raise ValueError('A pending IF requires its complete raw call')
                    try:
                        pending.explicit_decision(if_calls[0]['raw'])
                    except pending.DecisionPending as exc:
                        if vote['IF'] != pending.pending(str(exc)):
                            raise ValueError('Pending IF reason mismatch')
                    else:
                        raise ValueError('An unambiguous IF cannot be pending')
                elif vote['IF']['status'] == 'response_incomplete':
                    if len(if_calls) != 1 or vote['IF'] != pending.incomplete_result(if_calls[0]):
                        raise ValueError('Missing or mismatched incomplete IF evidence')
            for call in if_calls:
                if call['image_hashes'] not in [sample['pixel_hashes'][turn - 1:turn + 1], sample['pixel_hashes'][turn:turn + 1]]:
                    raise ValueError('IF image pair mismatch')
            if any(pending.incomplete_trace(c) for c in if_calls) != (vote['IF']['status'] == 'response_incomplete'):
                raise ValueError('Incomplete IF must remain unresolved')
            for call in judgment['traces']:
                incomplete = pending.incomplete_trace(call)
                if ((not incomplete and (call.get('error') or not call['raw'].strip() or call['finish_reason'] != 'stop')) or
                        call['vote'] != 1 or call['sampling_seed'] != 42 or call['temperature'] != 0.6 or
                        Path(call['model']).name != 'Qwen3.6-27B' or
                        call['request_hash'] != core.request_identity(call['image_hashes'], call['prompt'], 1) or
                        call['prompt_tokens'] + call['completion_tokens'] > call['runtime_context_limit'] or
                        call['runtime_context_limit'] != 16384):
                    raise ValueError('Raw Qwen request identity or completion failure')
                if incomplete:
                    from .scoring.request_cache import RequestCache
                    pin = next(x for x in manifest['models']['models'] if x['name'] == 'Qwen3.6-27B')
                    attempts = pending.IncompleteResponseCache(output / 'response_attempts/qwen', call['model'], pin,
                                                               manifest['fingerprint'])
                    evidence = attempts.get(call['request_hash'])
                    if not evidence or evidence['trace']['raw'] != call['raw']:
                        raise ValueError('Incomplete response was not preserved immutably')
                    if RequestCache(output / 'requests/qwen', call['model'], pin, manifest['fingerprint']).get(call['request_hash']):
                        raise ValueError('An incomplete vote must not have a replacement answer')
            prefixes.append(judgment['GA_prefix'])
            running_ga = pending.cumulative(prefixes)[-1]
            reported = indexed[(sample['session_id'], turn)]
            if (reported['IF_qwen'] != judgment['IF'] or reported['CC'] != metric['CC'] or
                    reported['GA_qwen'] != running_ga or
                    reported['GA_prefix_qwen'] != judgment['GA_prefix']):
                raise ValueError('Reported score mismatch')
            traces += len(judgment['traces'])
            expected_pending.extend(pending.pending_entries(judgment))
    review = core.read(output / 'pending_review.json')
    if (review['items'] != expected_pending or review['count'] != len(expected_pending) or
            summary['pending_components'] != len(expected_pending)):
        raise ValueError('Manual review index is incomplete')
    value = {'status': 'passed', 'protocol': PROTOCOL, 'judges': ['qwen'], 'votes_per_judge': 1,
             'sessions': len(manifest['records']), 'turns': len(scores), 'judge_requests': traces,
             'pending_components': len(expected_pending), 'decision_policy': pending.POLICY,
             'incomplete_response_components': sum(v['status'] == 'response_incomplete' for v in expected_pending),
             'scope': 'actual engineering evidence; human accuracy not calibrated',
             'checks': ['source and input hashes', 'every result and detection checksum',
                        'exactly one Qwen vote', 'IF previous/current image identity',
                        'complete GA prefixes and exact prompts', 'raw answers, seeds and token limits',
                        'cumulative GA and reported values', 'full session/turn coverage',
                        'pending decisions preserve complete raw evidence and manual review index']}
    core.write(output / 'validation.json', value)
    print(json.dumps(value), flush=True)


def main():
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'metrics', 'judge', 'report', 'validate'])
    parser.add_argument('--dataset', type=Path, default=Path('/home/chs/dataset/MICE-Bench'))
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, default=Path('/home/chs/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-metrics', type=Path)
    parser.add_argument('--allow-full', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    args.output, args.inference = args.output.resolve(), args.inference.resolve()
    args.selection, args.judge = 'all', 'qwen'
    if not args.allow_full or args.output.is_relative_to(args.inference) or args.inference.is_relative_to(args.output):
        raise ValueError('Explicit --allow-full and separate output are required')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'stage': args.stage, 'judge': 'qwen' if args.stage == 'judge' else None,
                  'status': 'running', 'started_at': time.time(), 'protocol': PROTOCOL}
        path = args.output / ('status_' + args.stage + ('_qwen' if args.stage == 'judge' else '') + '.json')
        core.write(path, status)
        try:
            if args.stage == 'prepare':
                runner.prepare(args)
                if args.reuse_metrics:
                    import_metrics(args.reuse_metrics, args.output)
            else:
                manifest = runner.load_manifest(args)
                if manifest['protocol'] != PROTOCOL:
                    raise ValueError('Wrong scoring profile')
                if args.stage == 'metrics': runner.metrics_stage(args, manifest)
                elif args.stage == 'judge': pending.judge_stage(args, manifest)
                elif args.stage == 'report': report(args, manifest)
                else: audit(args.output, manifest)
            status['status'] = 'complete'
        except Exception as exc:
            status.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            status['finished_at'] = time.time()
            core.write(path, status)

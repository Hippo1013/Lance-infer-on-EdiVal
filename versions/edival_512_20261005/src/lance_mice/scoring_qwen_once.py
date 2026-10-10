"""Independent Qwen-only, one-vote profile over the unchanged scoring primitives."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

from .scoring import core, runner

PROTOCOL = 'mice-qwen-once-v1'
BASE_SOURCE_PINS = core.source_pins
PROFILE_FILES = ('src/lance_mice/scoring_qwen_once.py', 'scripts/score_mice_qwen_once.py',
                 'scripts/run_mice_qwen_scoring.sh', 'scripts/full_mice_qwen_once_job.py',
                 'scripts/full_mice_scoring_job.py')


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
        if core.digest(old) != old_fp or old['protocol'] != 'mice-dual-judge-v2':
            raise ValueError('Unsupported or changed producer manifest')
        if not (source / 'superseded.json').exists() or not (source / 'completion.json').exists():
            raise ValueError('Producer must have stopped before measurement reuse')
        if old['records'] != new['records'] or old['dataset_hashes'] != new['dataset_hashes']:
            raise ValueError('Metric inputs changed')
        if old['models'] != new['models']:
            raise ValueError('Model revisions changed')
        if any(new['sources'].get(name) != pin for name, pin in old['sources'].items()):
            raise ValueError('The original scoring primitives changed')
        settings = dict(old['settings'])
        if settings['votes_per_judge'] != 2:
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
        value = {'status': 'passed', 'source': str(source), 'source_fingerprint': old_fp,
                 'source_manifest_sha256': core.sha(source / 'manifest.json'),
                 'fingerprint': new['fingerprint'], 'metrics_reused': metrics,
                 'detections_reused': detections, 'judge_answers_reused': 0,
                 'checks': ['stopped producer and exclusive lock', 'identical original scoring source pins',
                            'identical images, metadata and model revisions', 'every reused result checksum',
                            'detection identities and dependencies', 'per-record producer file hash']}
        core.write(output / 'metric_reuse.json', value)
        print(json.dumps(value), flush=True)


def report(args, manifest):
    rows, nonmonotonic = [], []
    for sample in manifest['records']:
        judgments = []
        for turn in range(1, 4):
            value = runner.saved(runner.artifact_path(args.output, 'qwen', sample, turn), manifest['fingerprint'])
            if value is None or value['status'] != 'ok':
                raise ValueError('Missing Qwen result')
            judgments.append(value)
        cumulative = core.cumulative([r['GA_prefix'] for r in judgments])
        for turn, judgment in enumerate(judgments, 1):
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
            if cumulative[turn - 1]['score'] < judgment['GA_prefix']['score']:
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
               'human_calibration': 'pending',
               'claim': 'Project Qwen single-vote scores; not official-paper or dual-judge scores.'}
    core.write(args.output / 'scores.json', rows)
    core.write(args.output / 'summary.json', summary)
    runner.build_review(args.output, manifest, rows)
    review = args.output / 'review.html'
    review.write_text(review.read_text().replace('MICE 小样本裁判核对', 'MICE Qwen 单次全量评分'))
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
    traces = 0
    for sample in manifest['records']:
        running_ga = 1.0
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
            if core.parse_ga(call['raw'], turn) != vote['GA_prefix']['score']:
                raise ValueError('Raw GA parsing mismatch')
            if judgment['GA_prefix'] != core.vote_mean([vote['GA_prefix']]):
                raise ValueError('Single-vote GA mismatch')
            if_calls = [c for c in judgment['traces'] if c['kind'] == 'IF']
            if 'shared_if' in metric:
                if if_calls or judgment['IF'] != metric['shared_if'] or vote['IF'] != metric['shared_if']:
                    raise ValueError('Shared IF mismatch')
            else:
                if [c['vote'] for c in if_calls] != [1] or judgment['IF'] != core.vote_mean([vote['IF']]):
                    raise ValueError('Single-vote IF mismatch')
            for call in if_calls:
                if call['image_hashes'] not in [sample['pixel_hashes'][turn - 1:turn + 1], sample['pixel_hashes'][turn:turn + 1]]:
                    raise ValueError('IF image pair mismatch')
            for call in judgment['traces']:
                if (call.get('error') or not call['raw'].strip() or call['finish_reason'] != 'stop' or
                        call['vote'] != 1 or call['sampling_seed'] != 42 or call['temperature'] != 0.6 or
                        Path(call['model']).name != 'Qwen3.6-27B' or
                        call['request_hash'] != core.request_identity(call['image_hashes'], call['prompt'], 1) or
                        call['prompt_tokens'] + call['completion_tokens'] > call['runtime_context_limit'] or
                        call['runtime_context_limit'] != 16384):
                    raise ValueError('Raw Qwen request identity or completion failure')
            running_ga = min(running_ga, judgment['GA_prefix']['score'])
            reported = indexed[(sample['session_id'], turn)]
            if (reported['IF_qwen'] != judgment['IF'] or reported['CC'] != metric['CC'] or
                    reported['GA_qwen']['score'] != running_ga or
                    reported['GA_prefix_qwen'] != judgment['GA_prefix']):
                raise ValueError('Reported score mismatch')
            traces += len(judgment['traces'])
    value = {'status': 'passed', 'protocol': PROTOCOL, 'judges': ['qwen'], 'votes_per_judge': 1,
             'sessions': len(manifest['records']), 'turns': len(scores), 'judge_requests': traces,
             'scope': 'actual engineering evidence; human accuracy not calibrated',
             'checks': ['source and input hashes', 'every result and detection checksum',
                        'exactly one Qwen vote', 'IF previous/current image identity',
                        'complete GA prefixes and exact prompts', 'raw answers, seeds and token limits',
                        'cumulative GA and reported values', 'full session/turn coverage']}
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
                elif args.stage == 'judge': runner.judge_stage(args, manifest)
                elif args.stage == 'report': report(args, manifest)
                else: audit(args.output, manifest)
            status['status'] = 'complete'
        except Exception as exc:
            status.update(status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            status['finished_at'] = time.time()
            core.write(path, status)

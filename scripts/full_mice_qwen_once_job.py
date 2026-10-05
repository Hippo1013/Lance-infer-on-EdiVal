#!/usr/bin/env python3
"""Single-GPU Qwen pass with measurement reuse and exit-driven stage transitions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from full_mice_scoring_job import now, write, release_burn, restore_burn

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/chs/conda/envs/lance/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-metrics', type=Path, required=True)
    parser.add_argument('--allow-full', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.allow_full or (output.exists() and any(output.iterdir())):
        raise ValueError('Explicit full authorization and a new output directory are required')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'state': 'starting', 'started_at': now(), 'pid': os.getpid(),
              'output': str(output), 'inference': str(args.inference.resolve()),
              'protocol': 'mice-qwen-once-v3', 'judges': ['qwen'], 'votes_per_judge': 1,
              'decision_policy': 'completed-decision-pending-review-v1',
              'incomplete_response_policy': 'judge-response-incomplete-v1',
              'temperature': 0.6, 'seed': 42, 'gpu': 0,
              'total_sessions': 720, 'total_turns': 2160, 'stages': {},
              'followup': 'User reviews this pass before authorizing further scoring'}
    sequence = 0
    child = None
    used_gpu = False

    def emit(event, **details):
        nonlocal sequence
        sequence += 1
        entry = {'sequence': sequence, 'time': now(), 'event': event, **details}
        with (output / 'events.jsonl').open('a') as log:
            log.write(json.dumps(entry, ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
        status.update(updated_at=entry['time'], last_event=entry)
        write(output / 'job_status.json', status)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def run(stage, command):
        nonlocal child
        with (output / f'{stage}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status['stages'][stage] = {'state': 'running', 'pid': child.pid,
                                      'started_at': now(), 'command': command}
            emit('stage_started', stage=stage, pid=child.pid)
            code = child.wait()
        status['stages'][stage].update(state='complete' if code == 0 else 'failed',
                                      exit_code=code, finished_at=now())
        emit('stage_completed' if code == 0 else 'stage_failed', stage=stage, exit_code=code)
        if code:
            raise RuntimeError(f'{stage} failed with exit code {code}; see {stage}.log')

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    common = ['--output', str(output), '--inference', str(args.inference.resolve()), '--allow-full']
    try:
        emit('job_started')
        run('prepare', [PYTHON, 'scripts/score_mice_qwen_once.py', 'prepare', *common,
                        '--reuse-metrics', str(args.reuse_metrics.resolve())])
        manifest = json.loads((output / 'manifest.json').read_text())
        if (manifest['protocol'] != status['protocol'] or manifest['settings']['votes_per_judge'] != 1 or
                manifest['settings']['temperature'] != 0.6 or len(manifest['records']) != 720):
            raise ValueError('Incorrect prepared profile')
        pins = manifest['sources']
        write(output / 'job_source_hashes.json', pins)
        status['fingerprint'] = manifest['fingerprint']
        reuse = json.loads((output / 'metric_reuse.json').read_text())
        emit('measurements_imported', **reuse)
        for stage in ('metrics', 'qwen'):
            if stage == 'metrics' and reuse['metrics_reused'] == status['total_turns']:
                status['stages']['metrics'] = {'state': 'complete', 'method': 'verified reuse',
                                             'completed_turns': reuse['metrics_reused']}
                emit('stage_reused', stage='metrics', completed_turns=reuse['metrics_reused'])
                continue
            status['state'] = stage
            emit('gpu_release_started', gpu=0)
            used_gpu = True
            release_burn(0)
            run(stage, ['bash', 'scripts/run_mice_qwen_scoring.sh', '0', stage, *common])
        status['state'] = 'report'
        for stage in ('report', 'validate'):
            run(stage, [PYTHON, 'scripts/score_mice_qwen_once.py', stage, *common])
        for name, expected in pins.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed: ' + name)
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        emit('job_error', error=status['error'])
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        time.sleep(10)
        if used_gpu:
            try:
                status['burn_restore'] = {'0': restore_burn(0)}
            except Exception as exc:
                status['burn_restore'] = {'0': 'error: ' + str(exc)}
        status.update(finished_at=now(), completed_turns={name: len(list((output / name).glob('*/*/turn_[1-3].json')))
                                                       for name in ('metrics', 'qwen')})
        write(output / 'completion.json', {'exit_code': 0 if status['state'] == 'completed' else 1, **status})
        emit('job_completed' if status['state'] == 'completed' else 'job_failed', state=status['state'])
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

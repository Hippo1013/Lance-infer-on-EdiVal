#!/usr/bin/env python3
"""Run a frozen full scoring pass; advance on child exits and record events."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
from threading import Thread
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/chs/conda/envs/lance/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    pending = path.with_suffix(path.suffix + '.part')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(pending, path)


def gpu_state(gpu):
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    pane = f'{gpu}:0.0'
    pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane,
                                  '#{pane_pid}'], text=True).strip()
    children = subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True)
    if children.returncode not in (0, 1):
        raise RuntimeError(f'Cannot inspect burn pane {pane}')
    commands = [subprocess.check_output(['ps', '-p', p, '-o', 'args='], text=True).strip()
                for p in children.stdout.split()]
    return memory, pane, commands


def release_burn(gpu):
    memory, pane, commands = gpu_state(gpu)
    if commands:
        if len(commands) != 1 or not commands[0].endswith(f'llamafactory_burn.sh {gpu}'):
            raise RuntimeError(f'Refusing to interrupt non-burn child in {pane}: {commands}')
        subprocess.run(['tmux', 'send-keys', '-t', pane, 'C-c'], check=True)
        time.sleep(25)
    memory, _, commands = gpu_state(gpu)
    if memory > 64 or commands:
        raise RuntimeError(f'GPU {gpu} not released: {memory} MiB, pane children={commands}')


def restore_burn(gpu):
    memory, pane, commands = gpu_state(gpu)
    if commands:
        if len(commands) == 1 and commands[0].endswith(f'llamafactory_burn.sh {gpu}'):
            return 'existing burn retained'
        return 'skipped: pane occupied'
    if memory > 64:
        return 'skipped: GPU occupied'
    command = ('cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 '
               'BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 '
               f'bash ./llamafactory_burn.sh {gpu}')
    subprocess.run(['tmux', 'send-keys', '-t', pane, command, 'C-m'], check=True)
    return 'start sent to idle burn pane'


def progress(output):
    return {name: len(list((output / name).glob('*/*/turn_[1-3].json')))
            for name in ('metrics', 'qwen', 'gemma')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--allow-full', action='store_true')
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.status_only:
        status = json.loads((output / 'job_status.json').read_text())
        print(json.dumps({**status, 'completed_turns': progress(output)}, ensure_ascii=False))
        return 0
    if not args.allow_full:
        raise ValueError('Full scoring requires explicit --allow-full')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('A new scoring output directory is required')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'state': 'starting', 'started_at': now(), 'pid': os.getpid(),
              'output': str(output), 'inference': str(args.inference.resolve()),
              'protocol': 'mice-dual-judge-v2', 'temperature': 0.6,
              'votes_per_judge': 2, 'vote_seeds': [42, 43],
              'total_sessions': 720, 'total_turns': 2160, 'stages': {}}
    children = {}
    handles = []
    released = set()
    sequence = 0

    def emit(event, **details):
        nonlocal sequence
        sequence += 1
        entry = {'sequence': sequence, 'time': now(), 'event': event, **details}
        with (output / 'events.jsonl').open('a') as log:
            log.write(json.dumps(entry, ensure_ascii=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
        status['updated_at'] = entry['time']
        status['last_event'] = entry
        write(output / 'job_status.json', status)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def spawn(name, command):
        log = (output / f'{name}.log').open('a')
        handles.append(log)
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        children[name] = child
        status['stages'][name] = {'state': 'running', 'pid': child.pid,
                                 'started_at': now(), 'command': command}
        emit('stage_started', stage=name, pid=child.pid)
        return child

    def finish(name, code):
        status['stages'][name].update(state='complete' if code == 0 else 'failed',
                                     exit_code=code, finished_at=now())
        emit('stage_completed' if code == 0 else 'stage_failed', stage=name,
             exit_code=code, completed_turns=progress(output))

    def run(name, command):
        code = spawn(name, command).wait()
        finish(name, code)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; see {name}.log')

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    common = ['--output', str(output), '--inference', str(args.inference.resolve()), '--allow-full']
    try:
        emit('job_started')
        run('prepare', [PYTHON, 'scripts/score_mice.py', 'prepare', '--selection', 'all', *common])
        manifest = json.loads((output / 'manifest.json').read_text())
        if (manifest['protocol'] != status['protocol'] or manifest['settings']['temperature'] != 0.6
                or manifest['settings']['votes_per_judge'] != 2 or len(manifest['records']) != 720):
            raise ValueError('Prepared manifest differs from authorized scoring protocol')
        pins = dict(manifest['sources'])
        for name in ('scripts/full_mice_scoring_job.py', 'scripts/prepare_scoring_judge_worker.py',
                     'scripts/validate_scoring_run.py'):
            pins[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        write(output / 'job_source_hashes.json', pins)
        status['fingerprint'] = manifest['fingerprint']
        status['state'] = 'metrics'
        emit('gpu_release_started', gpu=0)
        released.add(0)
        release_burn(0)
        run('metrics', ['bash', 'scripts/run_mice_scoring.sh', '0', 'metrics', *common])
        run('worker', [PYTHON, 'scripts/prepare_scoring_judge_worker.py',
                       '--output', str(output), '--judge', 'gemma'])
        status['state'] = 'judges'
        # Stop only the known burn children; the scoring wrappers restore each card.
        for gpu in (0, 1):
            emit('gpu_release_started', gpu=gpu)
            released.add(gpu)
            release_burn(gpu)
        spawn('qwen', ['bash', 'scripts/run_mice_scoring.sh', '0', 'qwen', *common])
        spawn('gemma', ['bash', 'scripts/run_mice_scoring.sh', '1', 'gemma',
                       '--output', str(output / 'worker_gemma'), '--allow-full'])
        failed = []
        completed = Queue()
        def wait_for_judge(name):
            completed.put((name, children[name].wait()))
        for name in ('qwen', 'gemma'):
            Thread(target=wait_for_judge, args=(name,), daemon=True).start()
        for _ in range(2):
            name, code = completed.get()
            finish(name, code)
            if code:
                failed.append(name)
        if failed:
            raise RuntimeError(f'Judge stages failed: {failed}; full raw evidence retained')
        status['state'] = 'report'
        run('report', [PYTHON, 'scripts/score_mice.py', 'report', *common])
        run('validation', [PYTHON, 'scripts/validate_scoring_run.py', '--output', str(output)])
        for name, expected in pins.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Source changed during scoring: {name}')
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        emit('job_error', error=status['error'])
        # Only stop process groups created by this job, never other server sessions.
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    finally:
        for handle in handles:
            handle.close()
        time.sleep(10)
        restore = {}
        for gpu in sorted(released):
            try:
                restore[str(gpu)] = restore_burn(gpu)
            except Exception as exc:
                restore[str(gpu)] = f'error: {exc}'
        status.update(finished_at=now(), burn_restore=restore, completed_turns=progress(output))
        write(output / 'completion.json', {'exit_code': 0 if status['state'] == 'completed' else 1,
                                          **status})
        emit('job_completed' if status['state'] == 'completed' else 'job_failed',
             state=status['state'], burn_restore=restore)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

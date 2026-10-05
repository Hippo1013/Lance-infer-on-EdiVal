#!/usr/bin/env python3
"""One background MICE pass, atomic status, final validation, and burn restoration."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from lance_mice.acceptance import validate_run
from lance_mice.dataset import DEFAULT_ROOTS, load_samples
from lance_mice.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(run):
    rows = list(run.glob('*/*/turn_[1-3].json'))
    return {'completed_turns': len(rows),
            'completed_sessions': sum((p.parent / 'turn_3.json').is_file() for p in rows if p.name == 'turn_1.json'),
            'attention_files': len(list(run.glob('*/*/turn_*.attention.npz')))}


def restore_burn():
    """Only idle GPU 0/1 and their existing shell panes; never other sessions."""
    result = {}
    for gpu in (0, 1):
        memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory > 64:
            result[str(gpu)] = 'skipped: GPU still occupied'
            continue
        pane = f'{gpu}:0.0'
        pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_pid}'], text=True).strip()
        children = subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True)
        if children.returncode != 1:
            result[str(gpu)] = 'skipped: pane has a running child or cannot be checked'
            continue
        command = ('cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 '
                   'BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 '
                   f'bash ./llamafactory_burn.sh {gpu}')
        subprocess.run(['tmux', 'send-keys', '-t', pane, command, 'C-m'], check=True)
        result[str(gpu)] = 'start sent to idle burn pane'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path, default=ROOT/'outputs/acceptance/bare_20261004_v2')
    p.add_argument('--restore-burn', action='store_true')
    p.add_argument('--status-only', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    run = output / 'run'
    if args.status_only:
        status = json.loads((output/'status.json').read_text())
        print(json.dumps({**status, **progress(run)}, ensure_ascii=False, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new full-pass directory')
    accepted = json.loads((args.acceptance/'summary.json').read_text())
    if accepted['status'] != 'passed' or not accepted['comparison']['identical_pixels']:
        raise ValueError('Attention engineering acceptance is required')
    if accepted.get('protocol') != 'lance-history-bare-v2':
        raise ValueError('Bare protocol acceptance is required')
    source_hashes = json.loads((args.acceptance/'source_hashes.json').read_text())
    for name, expected in source_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source differs from accepted version: {name}')
    samples = load_samples(DEFAULT_ROOTS['mice'], 'all')
    if len(samples) != 720 or sum(len(s.instructions) for s in samples) != 2160:
        raise ValueError('Unexpected MICE release size')
    status = {'state':'starting', 'started_at':now(), 'pid':os.getpid(), 'total_sessions':720,
              'total_turns':2160, 'profile':'dp2', 'cache':'prefix', 'attention':'target-group-mass-v1',
              'output':str(run), 'capability_scoring':'not run', 'attempts_per_session':1,
              'protocol':accepted['protocol']}
    write_json(output/'source_hashes.json', source_hashes)
    write_json(output/'status.json', status)
    cmd = [sys.executable, '-m', 'lance_mice.runner', '--selection','all','--profile','dp2',
           '--gpus','0,1','--cache-mode','prefix','--audit','--save-attention','--output',str(run)]
    status['command'] = cmd
    child = None
    start = time.monotonic()
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (output/'run.log').open('w') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(state='running', runner_pid=child.pid)
            while child.poll() is None:
                status.update(updated_at=now(), elapsed_seconds=time.monotonic()-start, **progress(run))
                write_json(output/'status.json', status)
                time.sleep(20)
            if child.returncode:
                raise RuntimeError(f'Runner exited with {child.returncode}; see run.log')
        status.update(state='validating', updated_at=now(), **progress(run))
        write_json(output/'status.json', status)
        from validate_mice_full import audit
        write_json(output/'validation.json', audit(output, DEFAULT_ROOTS['mice']))
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        status.update(updated_at=now(), finished_at=now(), elapsed_seconds=time.monotonic()-start, **progress(run))
        write_json(output/'status.json', status)
        if args.restore_burn:
            time.sleep(10)
            try:
                status['burn_restore'] = restore_burn()
            except Exception as exc:
                status['burn_restore'] = {'error':str(exc)}
            write_json(output/'status.json', status)
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

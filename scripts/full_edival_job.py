#!/usr/bin/env python3
"""One complete single-GPU EdiVal pass, atomic progress and final CPU provenance audit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lance_mice.dataset import DEFAULT_ROOTS, load_samples
from lance_mice.edival_dataset import source_identity
from lance_mice.runner import write_json
from edival_acceptance import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(run):
    rows = list(run.glob('edival/*/turn_[1-3].json'))
    return {'completed_turns': len(rows),
            'completed_sessions': len(list(run.glob('edival/*/turn_3.json'))),
            'attention_files': len(list(run.glob('edival/*/turn_*.attention.npz')))}


def restore_burn(gpu):
    """Restore only this run's physical GPU and its existing idle burn pane."""
    if gpu not in ('0', '1'):
        raise ValueError('Invalid physical GPU')
    memory = int(subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    pane = f'{gpu}:0.0'
    pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane, '#{pane_pid}'], text=True).strip()
    children = subprocess.run(['pgrep', '-P', pid], capture_output=True, text=True)
    if memory <= 64 and children.returncode == 1:
        command = ('cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 '
            'BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 '
            f'bash ./llamafactory_burn.sh {gpu}')
        subprocess.run(['tmux', 'send-keys', '-t', pane, command, 'C-m'], check=True)
        return {'gpu': gpu, 'status': 'burn_start_sent', 'memory_before_restore_mib': memory}
    return {'gpu': gpu, 'status': 'skipped_gpu_or_pane_busy', 'memory_mib': memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('0', '1'), default='1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_ROOTS['edival'])
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    output, run = args.output.resolve(), args.output.resolve() / 'run'
    if args.status_only:
        print(json.dumps({**json.loads((output / 'status.json').read_text()), **progress(run)}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    status = {'state': 'preflight', 'started_at': now(), 'pid': os.getpid(),
              'gpu': args.gpu, 'profile': 'single', 'cache': 'prefix', 'resolution': 512,
              'protocol': 'lance-history-bare-v2', 'total_sessions': 572, 'total_turns': 1716,
              'attention': 'target-group-mass-v1', 'capability_scoring': 'not run',
              'attempts_per_session': 1, 'runtime_source': str(ROOT), 'output': str(run)}
    write_json(output / 'status.json', status)
    child, start = None, time.monotonic()

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        accepted = json.loads(args.acceptance.read_text())
        if (accepted['status'] != 'passed' or accepted.get('resolution') != 512
                or not accepted['comparison']['identical_pixels'] or accepted['attention_files'] != 6):
            raise ValueError('512-pixel cache and attention acceptance is required')
        pins = source_hashes()
        if pins != accepted['source_hashes']:
            raise ValueError('Source differs from 512-pixel acceptance')
        if source_identity(args.dataset) != accepted['dataset_source']:
            raise ValueError('Dataset differs from accepted release')
        samples = load_samples(args.dataset, 'all', 'edival')
        if len(samples) != 572 or sum(len(s.instructions) for s in samples) != 1716:
            raise ValueError('Unexpected EdiVal release coverage')
        write_json(output / 'source_hashes.json', pins)
        write_json(output / 'launch.json', {'acceptance': str(args.acceptance), 'source_hashes': pins,
                   'dataset_source': accepted['dataset_source'], 'samples': [s.session_id for s in samples]})
        command = [sys.executable, '-B', '-m', 'lance_mice.edival', '--selection', 'all',
                   '--profile', 'single', '--gpus', args.gpu, '--resolution', '512',
                   '--dataset', str(args.dataset), '--cache-mode', 'prefix', '--audit',
                   '--save-attention', '--output', str(run)]
        status['command'] = command
        with (output / 'run.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(state='running', runner_pid=child.pid)
            while child.poll() is None:
                status.update(updated_at=now(), elapsed_seconds=time.monotonic() - start, **progress(run))
                write_json(output / 'status.json', status)
                print(json.dumps({k: status[k] for k in ('state', 'completed_turns', 'completed_sessions', 'attention_files')}), flush=True)
                time.sleep(20)
            if child.returncode:
                raise RuntimeError(f'Inference exited with {child.returncode}; see run.log')
        status.update(state='validating', updated_at=now(), **progress(run))
        write_json(output / 'status.json', status)
        from validate_edival_full import audit
        write_json(output / 'validation.json', audit(output, args.dataset))
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
    finally:
        status.update(updated_at=now(), finished_at=now(), elapsed_seconds=time.monotonic() - start, **progress(run))
        write_json(output / 'status.json', status)
        write_json(output / 'completion.json', {'state': status['state'], 'exit_code': 0 if status['state'] == 'completed' else 1,
                    'finished_at': status['finished_at'], 'error': status.get('error')})
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

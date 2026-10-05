#!/usr/bin/env python3
"""One background ImgEdit pass, atomic status, final validation, and burn restoration."""
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
from lance_mice.dataset import DEFAULT_ROOTS, load_samples, shard_samples
from lance_mice.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(run):
    manifest = run / 'run.json'
    samples = json.loads(manifest.read_text())['samples'] if manifest.exists() else []
    return {'completed_turns': len(list(run.glob('*/*/turn_[1-3].json'))),
            'completed_sessions': sum((run / sid / f'turn_{len(instructions)}.json').is_file()
                                      for sid, instructions in samples),
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
    imgedit_acceptance = json.loads((args.acceptance/'imgedit_summary.json').read_text())
    if imgedit_acceptance['status'] != 'passed':
        raise ValueError('ImgEdit engineering acceptance is required')
    samples = load_samples(DEFAULT_ROOTS['imgedit'], 'all', 'imgedit')
    if len(samples) != 30 or sum(len(s.instructions) for s in samples) != 88:
        raise ValueError('Unexpected ImgEdit release size')
    status = {'state':'starting', 'started_at':now(), 'pid':os.getpid(), 'total_sessions':30,
              'total_turns':88, 'profile':'dp2', 'cache':'prefix', 'attention':'target-group-mass-v1',
              'output':str(run), 'capability_scoring':'not run', 'attempts_per_session':1,
              'protocol':accepted['protocol']}
    source_hashes['scripts/full_imgedit_job.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output/'source_hashes.json', source_hashes)
    write_json(output/'status.json', status)
    cmd = [sys.executable, '-m', 'lance_mice.imgedit', '--selection','all','--profile','dp2',
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
        rows = validate_run(run)
        manifest = json.loads((run/'run.json').read_text())
        expected_samples = [[s.session_id, list(s.instructions)] for s in samples]
        if manifest['samples'] != expected_samples:
            raise ValueError('Dataset session/instruction manifest differs')
        from PIL import Image
        from lance_mice.images import image_hash
        for sample in samples:
            with Image.open(sample.image) as original, Image.open(run/sample.session_id/'turn_0_input.png') as saved:
                if image_hash(original) != image_hash(saved):
                    raise ValueError(f'Source image differs: {sample.session_id}')
        worker_pids = []
        for i in range(2):
            worker = json.loads((run/f'worker_{i}.json').read_text())
            assigned = shard_samples(samples, i, 2)
            expected = [[s.session_id, t] for s in assigned for t in range(1, len(s.instructions)+1)]
            if not worker['completed'] or worker['generated_turns'] != expected or worker['gpu'] != str(i):
                raise ValueError(f'Worker {i} assignment/completion differs')
            worker_pids.append(worker['pid'])
        if len(set(worker_pids)) != 2:
            raise ValueError('Workers must be separate processes')
        for name, expected in source_hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Source changed during inference: {name}')
        if len(rows) != 88:
            raise ValueError('Full pass contains missing turns')
        write_json(output/'validation.json', {'status':'passed','sessions':30,'turns':len(rows),
            'attention_files':len(rows), 'max_reference_absolute_error':max(
                x['backend']['attention']['max_reference_absolute_error'] for x in rows.values()),
            'scope':'Raw instructions, full model-generated history, CFG, seeds, image hashes and attention artifacts. No capability scoring.'})
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

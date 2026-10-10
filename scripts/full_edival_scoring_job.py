#!/usr/bin/env python3
"""Two independent GPUs, session shards, three sequential official score stages."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from edival_scoring import ROOT, STAGES, checked_stage, now, read, sha, write

ENVS = {'if': 'EdiVal-judge', 'metrics': 'EdiVal', 'hps': 'EdiVal-hps'}


def memory(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True).strip())


def idle_pane(gpu):
    pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', f'{gpu}:0.0', '#{pane_pid}'], text=True).strip()
    return subprocess.run(['pgrep', '-P', pid], capture_output=True).returncode == 1


def wait_free(gpu):
    for _ in range(45):
        if memory(gpu) <= 64:
            return
        time.sleep(1)
    raise RuntimeError(f'GPU{gpu} occupied during phase transition: {memory(gpu)} MiB')


def env_for(gpu, stage, temporary):
    env = os.environ.copy()
    name = ENVS[stage]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda-13.0',
               LD_LIBRARY_PATH=f'/usr/local/cuda-13.0/compat:/home/chs/conda/envs/{name}/lib:' + env.get('LD_LIBRARY_PATH', ''),
               TMPDIR=temporary, XDG_CACHE_HOME=temporary+'/cache', VLLM_CACHE_ROOT=temporary+'/vllm',
               TORCHINDUCTOR_CACHE_DIR=temporary+'/inductor', TRITON_CACHE_DIR=temporary+'/triton',
               MPLCONFIGDIR=temporary+'/matplotlib', PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_USE_V2_MODEL_RUNNER='0', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               TOKENIZERS_PARALLELISM='false')
    return env


def restore_burn(gpu):
    # Never interrupt an unrelated process. Restore only the now-empty owned card.
    wait_free(gpu)
    if not idle_pane(gpu):
        return dict(gpu=gpu, status='skipped_pane_busy')
    command = ('cd /media/damoxing/tangzecong && BURN_STEPS=2000 BURN_BATCH=8 '
               'BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 '
               f'bash ./llamafactory_burn.sh {gpu}')
    subprocess.run(['tmux', 'send-keys', '-t', f'{gpu}:0.0', command, 'C-m'], check=True)
    return dict(gpu=gpu, status='burn_start_sent', at=now())


def run_gpu(args):
    """One owned GPU stays allocated to this runner across process environments."""
    gpu, output = args.gpu_runner, args.output.resolve()
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    temporary = tempfile.mkdtemp(prefix=f'umm-edival-score-gpu{gpu}.', dir='/tmp')
    status = dict(gpu=gpu, state='preflight', started_at=now(), pid=os.getpid())
    try:
        if not idle_pane(gpu) or memory(gpu) > 64:
            raise RuntimeError(f'GPU{gpu} has not been released from burn/real tasks')
        manifest = read(output / 'manifest.json')
        for stage in STAGES:
            wait_free(gpu)
            status.update(state='running', stage=stage, updated_at=now())
            write(output / f'gpu_{gpu}.json', status)
            command = [f'/home/chs/conda/envs/{ENVS[stage]}/bin/python', '-B',
                       str(ROOT / 'scripts/edival_scoring.py'), 'worker', '--output', str(output),
                       '--stage', stage, '--shard', str(gpu)]
            with (output / f'gpu_{gpu}_{stage}.log').open('x') as log:
                child = subprocess.Popen(command, env=env_for(gpu, stage, temporary), cwd=temporary,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status['child_pid'] = child.pid
                write(output / f'gpu_{gpu}.json', status)
                rc = child.wait()
                child = None
            if rc:
                raise RuntimeError(f'{stage} exited {rc}; see gpu_{gpu}_{stage}.log')
        status.update(state='completed', finished_at=now(), child_pid=None,
                      sessions=len(manifest['shards'][gpu]), turns=3*len(manifest['shards'][gpu]))
    except BaseException as exc:
        status.update(state='failed', finished_at=now(), error=f'{type(exc).__name__}: {exc}')
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                status['child_cleanup'] = 'timeout; preserve ownership, do not restore burn'
    finally:
        shutil.rmtree(temporary)
        if not args.keep_released and (child is None or child.poll() is not None):
            try:
                status['burn_restore'] = restore_burn(gpu)
            except Exception as exc:
                status['burn_restore'] = dict(status='failed', error=str(exc))
        write(output / f'gpu_{gpu}.json', status)
    return 0 if status['state'] == 'completed' else 1


def progress(output):
    data = {}
    for gpu in (0, 1):
        data[str(gpu)] = read(output / f'gpu_{gpu}.json') if (output / f'gpu_{gpu}.json').exists() else {}
    data['coverage'] = {s: len(list((output / 'stages' / s).glob('*.json')))*3 for s in STAGES}
    return data


def run(args):
    output = args.output.resolve()
    if (output / 'completion.json').exists() or (output / 'status.json').exists():
        raise ValueError('Run already started; no automatic restart or resampling')
    if read(output / 'preflight.json')['status'] != 'passed':
        raise ValueError('Passed CPU preflight required')
    manifest = read(output / 'manifest.json')
    for gpu in (0, 1):
        if not idle_pane(gpu) or memory(gpu) > 64:
            raise RuntimeError(f'GPU{gpu} is busy; release only its authorized burn first')
    status = dict(state='starting', started_at=now(), pid=os.getpid(), protocol=manifest['protocol'],
                  manifest_sha256=sha(output / 'manifest.json'), total_sessions=manifest['session_count'],
                  total_turns=manifest['turn_count'], gpu_mode='session-sharded replicas; TP=1', gpus=[0, 1])
    children = []
    locks = []
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for gpu in (0, 1):
            lock = open(output.parent / f'.edival-scoring-gpu{gpu}.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        write(output / 'status.json', status)
        for gpu in (0, 1):
            command = [sys.executable, '-B', __file__, '--output', str(output), '--gpu-runner', str(gpu)]
            if args.keep_released:
                command.append('--keep-released')
            log = (output / f'gpu_{gpu}_supervisor.log').open('x')
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            log.close()
        status.update(state='running', runner_pids=[p.pid for p in children])
        while any(p.poll() is None for p in children):
            status.update(updated_at=now(), progress=progress(output))
            write(output / 'status.json', status)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('A GPU shard failed; stop owned remaining runner and preserve evidence')
            time.sleep(15)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('A GPU shard failed')
        status.update(state='validating', progress=progress(output), updated_at=now())
        write(output / 'status.json', status)
        subprocess.run(['/home/chs/conda/envs/EdiVal/bin/python', '-B', str(ROOT / 'scripts/edival_scoring.py'),
                        'aggregate', '--output', str(output)], check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    status.setdefault('cleanup_timeouts', []).append(child.pid)
    finally:
        for lock in locks:
            lock.close()
        status.update(finished_at=now(), updated_at=now(), progress=progress(output))
        write(output / 'status.json', status)
        write(output / 'completion.json', dict(state=status['state'], exit_code=0 if status['state']=='completed' else 1,
                    error=status.get('error'), finished_at=status['finished_at'], full_release=manifest['session_count']==572))
    return 0 if status['state'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-runner', type=int, choices=[0, 1])
    parser.add_argument('--keep-released', action='store_true', help='Acceptance followed immediately by full run')
    parsed = parser.parse_args()
    raise SystemExit(run(parsed) if parsed.gpu_runner is None else run_gpu(parsed))

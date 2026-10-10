#!/usr/bin/env python3
"""Restart an owned worker after interruption; preserve all completed evidence."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from sixrun_queue import ROOT, queue_lock, read, write, now


def restore_burn(host, gpu):
    used = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    pid = subprocess.check_output(['tmux', 'display-message', '-p', '-t', f'{gpu}:0.0', '#{pane_pid}'], text=True).strip()
    if used > 64 or subprocess.run(['pgrep', '-P', pid], capture_output=True).returncode != 1:
        return 'skipped: GPU or pane occupied'
    script = ('/media/damoxing/tangzecong/llamafactory_burn.sh' if host == 'a800_0'
              else '/home/chs/tools/gpu-burn/llamafactory_burn.sh')
    command = ('BURN_STEPS=2000 BURN_BATCH=8 BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 '
               f'BURN_LORA_RANK=128 bash {script} {gpu}')
    subprocess.run(['tmux', 'send-keys', '-t', f'{gpu}:0.0', command, 'C-m'], check=True)
    return 'burn start sent'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--host', choices=['a800_0', 'a800_1'], required=True)
    p.add_argument('--gpu', choices=['0', '1'], required=True)
    a = p.parse_args()
    with (a.base / f'.supervisor_{a.host}_{a.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recover a dead prior supervisor's assignment, but never a live worker.
        with queue_lock(a.base):
            jobs = read(a.base / 'queue.json')
            for job in jobs:
                if (job['state'] == 'running' and job.get('owner_host') == a.host
                        and job.get('owner_gpu') == a.gpu):
                    try:
                        os.kill(job['owner_pid'], 0)
                    except ProcessLookupError:
                        job.update(state='failed', retry_after=time.time(), finished_at=now())
                        job.setdefault('errors', []).append({'at': now(), 'message': 'Prior owned worker exited unexpectedly'})
                    else:
                        raise RuntimeError('Owned worker is already alive; refuse duplicate')
            write(a.base / 'queue.json', jobs)
        attempt = 0
        while not (a.base / 'completion.json').exists():
            attempt += 1
            log_path = a.base / 'logs' / f'{a.host}_{a.gpu}_worker_{attempt}_{time.time_ns()}.log'
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open('a') as log:
                child = subprocess.Popen([sys.executable, '-B', str(ROOT / 'scripts/sixrun_queue.py'), 'worker',
                        '--base', str(a.base), '--host', a.host, '--gpu', a.gpu],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(a.base / 'supervisors' / f'{a.host}_{a.gpu}.json',
                      {'state': 'running', 'pid': os.getpid(), 'worker_pid': child.pid,
                       'host': a.host, 'gpu': a.gpu, 'at': now(), 'log': str(log_path)})
                rc = child.wait()
            if rc == 0 and (a.base / 'completion.json').exists():
                break
            # Kill only descendants in the process group created by this supervisor.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            with queue_lock(a.base):
                jobs = read(a.base / 'queue.json')
                for job in jobs:
                    if job['state'] == 'running' and job.get('owner_pid') == child.pid and job.get('owner_host') == a.host:
                        job.update(state='failed', retry_after=time.time()+30, finished_at=now())
                        job.setdefault('errors', []).append({'at': now(), 'message': f'Owned worker exited {rc}'})
                write(a.base / 'queue.json', jobs)
            time.sleep(30)
            used = int(subprocess.check_output(['nvidia-smi', '-i', a.gpu, '--query-gpu=memory.used',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if used > 64:
                write(a.base / 'supervisors' / f'{a.host}_{a.gpu}.json',
                      {'state': 'attention_required', 'at': now(), 'error': 'Owned GPU memory not released after failure', 'memory_mib': used})
                raise RuntimeError('GPU still occupied after cleanup; preserve ownership for inspection')
        time.sleep(5)
        write(a.base / 'supervisors' / f'{a.host}_{a.gpu}.json',
              {'state': 'completed', 'at': now(), 'burn': restore_burn(a.host, a.gpu)})


if __name__ == '__main__':
    main()

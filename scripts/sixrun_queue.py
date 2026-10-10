#!/usr/bin/env python3
"""Four independent GPUs, whole-session jobs, local attention and shared scoring."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

from lance_mice.dataset import DEFAULT_ROOTS, load_samples
from lance_mice.protocol import digest, PROTOCOL_VERSION, CHAT_PROTOCOL_VERSION
from lance_mice.runner import model_identity, run_session
from lance_mice.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'target-token-region-stats-v1'
PROTOS = {'bare': PROTOCOL_VERSION, 'chat': CHAT_PROTOCOL_VERSION}
ENVS = {'if': 'EdiVal-judge', 'metrics_edival': 'EdiVal', 'hps': 'EdiVal-hps',
        'metrics_mice': 'mice-metrics', 'judge': 'mice-judge'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Different writers never share the staging filename.
    part = path.with_name(path.name + f'.{socket.gethostname()}.{os.getpid()}.pending')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    part.replace(path)


@contextmanager
def directory_lock(path):
    """GPFS flock is host-local here; mkdir is atomic across both hosts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            # Recover a dead owner only when its process can be checked locally.
            owner_path = path / 'owner.json'
            if owner_path.exists():
                try:
                    owner = read(owner_path)
                except FileNotFoundError:
                    continue
                if owner['hostname'] == socket.gethostname():
                    try:
                        os.kill(owner['pid'], 0)
                    except ProcessLookupError:
                        recovery = path.parent / (path.name + '.dead.' + str(time.time_ns()))
                        try:
                            path.rename(recovery)
                        except FileNotFoundError:
                            pass
                        continue
            time.sleep(.05)
    token = {'hostname': socket.gethostname(), 'pid': os.getpid(), 'at': now(), 'nonce': time.time_ns()}
    write(path / 'owner.json', token)
    try:
        yield
    finally:
        if read(path / 'owner.json') != token:
            raise RuntimeError('Atomic directory lock ownership changed')
        (path / 'owner.json').unlink()
        path.rmdir()


@contextmanager
def queue_lock(base):
    with directory_lock(base / '.queue.lease'):
        yield


def initialize(a):
    a.base.mkdir(parents=True, exist_ok=False)
    pins = {str(p.relative_to(ROOT)): sha(p) for top in ['src', 'scripts', 'configs']
            for p in (ROOT / top).rglob('*') if p.is_file() and not p.name.startswith('._') and p.suffix in ['.py', '.sh', '.json']}
    jobs, runs = [], {}
    for bench in ['edival', 'mice', 'imgedit']:
        samples = load_samples(DEFAULT_ROOTS[bench], 'all', bench)
        expected = {'edival': (572, 1716), 'mice': (720, 2160), 'imgedit': (30, 88)}[bench]
        assert (len(samples), sum(len(s.instructions) for s in samples)) == expected
        for label, proto in PROTOS.items():
            name = f'{bench}_{label}'
            settings = Settings(resolution=512 if bench == 'edival' else 768, cache_mode='prefix',
                                history_protocol=proto, attention_format=FORMAT)
            spec = {'settings': settings.identity(), 'profile': 'dynamic-dp4', 'audit': True,
                    'model': '/home/chs/model/Lance', 'model_files': model_identity(Path('/home/chs/model/Lance')),
                    'selection': 'all', 'benchmark': bench, 'attention': FORMAT,
                    'samples': [(s.session_id, list(s.instructions)) for s in samples]}
            manifest = {'fingerprint': digest(spec), **spec}
            output = a.base / 'inference' / name
            write(output / 'run/run.json', manifest)
            write(output / 'source_hashes.json', pins)
            runs[name] = {'benchmark': bench, 'label': label, 'sessions': len(samples),
                          'turns': expected[1], 'fingerprint': manifest['fingerprint']}
            for sample in samples:
                jobs.append({'id': f'infer:{name}:{sample.session_id}', 'kind': 'infer', 'run': name,
                             'session': sample.session_id, 'turns': len(sample.instructions), 'state': 'pending'})
    write(a.base / 'plan.json', {'created_at': now(), 'runs': runs, 'source_hashes': pins,
          'attention': FORMAT, 'scoring': {'edival': 'official IF/CC/RAHF/HPS', 'mice': 'Qwen3.6-27B one vote'},
          'local_outputs': '/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007',
          'runtime': str(ROOT), 'expected_turns': 7928})
    write(a.base / 'queue.json', jobs)
    print('QUEUE_PREPARED', len(jobs), flush=True)


def recover_transaction(folder):
    """Keep every interrupted artifact; only roll back an uncommitted next turn."""
    if not folder.exists():
        return
    for turn in range(1, 4):
        files = [folder / f'turn_{turn}.{ext}' for ext in ['png', 'attention.npz', 'json']]
        existing = [p for p in files if p.exists()]
        parts = list(folder.glob(f'turn_{turn}.*.part'))
        if len(existing) in (0, 3) and not parts:
            continue
        if (folder / f'turn_{turn}.json').exists():
            raise ValueError('Committed metadata with incomplete transaction; manual integrity repair required')
        if any((folder / f'turn_{later}.json').exists() for later in range(turn+1, 4)):
            raise ValueError('Interrupted transaction precedes committed history')
        evidence = folder / 'interrupted_transactions' / f'{time.time_ns()}'
        evidence.mkdir(parents=True)
        for path in existing + parts:
            path.replace(evidence / path.name)
        write(evidence / 'reason.json', {'reason': 'No committed turn metadata; retain original interrupted files', 'at': now()})


def inference(a, job, backend, samples):
    from PIL import Image, PngImagePlugin
    from lance_mice.acceptance import validate_run
    PngImagePlugin.MAX_TEXT_CHUNK = 4 * 1024 * 1024
    run = job['run']
    canonical = a.base / 'inference' / run
    manifest = read(canonical / 'run/run.json')
    local = a.local / run / 'run'
    if not (local / 'run.json').exists():
        write(local / 'run.json', manifest)
    elif read(local / 'run.json') != manifest:
        raise ValueError('Local run identity changed')
    sample = samples[(manifest['benchmark'], job['session'])]
    backend.settings = Settings(**manifest['settings']['settings'])
    folder = local / sample.session_id
    recover_transaction(folder)
    # A process may have stopped between writing the session spec and its source PNG.
    if (folder / 'session.json').exists() and not (folder / 'turn_0_input.png').exists():
        from lance_mice.images import image_hash
        source = sample.read_source()
        if image_hash(source) != read(folder / 'session.json')['source_hash']:
            raise ValueError('Interrupted source initialization changed')
        source.save(folder / 'turn_0_input.png')
    run_session(sample, local, backend.settings, backend, run_id=manifest['fingerprint'],
                resume=True, save_attention=True)
    rows = validate_run(local, sessions=[(sample.session_id, list(sample.instructions))])
    spec = read(folder / 'session.json')
    from lance_mice.images import image_hash
    expected_fp = digest([sample.session_id, sample.instructions, image_hash(sample.read_source()),
                          manifest['settings'], manifest['fingerprint']])
    if spec['fingerprint'] != expected_fp or any(r['session_fingerprint'] != expected_fp for r in rows.values()):
        raise ValueError('Session fingerprint/source changed')
    destination = canonical / 'run' / sample.session_id
    destination.mkdir(parents=True, exist_ok=True)
    image_pins, attention_pins = {}, {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix in ['.png', '.json']:
            target = destination / path.name
            if target.exists() and sha(target) != sha(path):
                raise ValueError('Canonical artifact conflict')
            if not target.exists():
                part = target.with_name(target.name + '.part')
                shutil.copyfile(path, part)
                part.replace(target)
            image_pins[path.name] = sha(target)
        elif path.name.endswith('.attention.npz'):
            attention_pins[path.name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    receipt = {'status': 'passed', 'at': now(), 'host': a.host, 'gpu': a.gpu, 'session': sample.session_id,
               'fingerprint': manifest['fingerprint'], 'session_fingerprint': expected_fp,
               'turns': len(rows), 'image_metadata_hashes': image_pins, 'attention': attention_pins,
               'local_run': str(local), 'source_hashes': read(a.base / 'plan.json')['source_hashes']}
    write(canonical / 'receipts' / (sample.session_id.replace('/', '__') + '.json'), receipt)


def execute(command, gpu, env_name, log_path, temporary):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda-13.0',
        LD_LIBRARY_PATH=f'/usr/local/cuda-13.0/compat:/home/chs/conda/envs/{env_name}/lib',
        PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=temporary, XDG_CACHE_HOME=temporary+'/cache', VLLM_CACHE_ROOT=temporary+'/vllm',
        TORCHINDUCTOR_CACHE_DIR=temporary+'/inductor', TRITON_CACHE_DIR=temporary+'/triton',
        MPLCONFIGDIR=temporary+'/matplotlib', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_USE_V2_MODEL_RUNNER='0', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        subprocess.run([f'/home/chs/conda/envs/{env_name}/bin/python', '-B', *command],
                       env=env, cwd=temporary, stdout=log, stderr=subprocess.STDOUT, check=True)


def finalize_inference(base, run):
    from lance_mice.acceptance import validate_run
    from PIL import PngImagePlugin
    PngImagePlugin.MAX_TEXT_CHUNK = 4 * 1024 * 1024
    output = base / 'inference' / run
    manifest = read(output / 'run/run.json')
    # All local attention files were numerically validated on the producing host.
    # Revalidate shared RGB history and traces here, then bind every receipt/file.
    rows = validate_run(output / 'run', validate_attention_files=False)
    attention_count, size = 0, 0
    for sid, instructions in manifest['samples']:
        receipt = read(output / 'receipts' / (sid.replace('/', '__') + '.json'))
        if receipt['status'] != 'passed' or receipt['fingerprint'] != manifest['fingerprint'] or receipt['turns'] != len(instructions):
            raise ValueError('Session receipt coverage/identity mismatch')
        for name, pin in receipt['image_metadata_hashes'].items():
            if sha(output / 'run' / sid / name) != pin:
                raise ValueError('Shared artifact changed after local validation')
        for turn in range(1, len(instructions)+1):
            artifact = receipt['attention'][f'turn_{turn}.attention.npz']
            if artifact['sha256'] != rows[f'{sid}/turn_{turn}']['backend']['attention']['sha256']:
                raise ValueError('Local attention receipt identity mismatch')
            attention_count += 1
            size += artifact['bytes']
    write(output / 'validation.json', {'status': 'passed', 'at': now(), 'sessions': len(manifest['samples']),
          'turns': len(rows), 'attention_files': attention_count, 'attention_bytes': size,
          'checks': ['canonical full RGB/history/protocol/seed/GPU trace validation',
                     'all local attention numerical/hash/geometry validators', 'all session receipts and artifact hashes'],
          'attention_storage': 'producing-host local files indexed in receipts; no shared duplicate'})
    write(output / 'completion.json', {'state': 'completed', 'exit_code': 0, 'finished_at': now()})


def score(a, job, temporary):
    run = job['run']
    output, inference_root = a.base / 'scoring' / run, a.base / 'inference' / run
    if job['kind'] == 'prepare_score':
        if run.startswith('edival'):
            command = [str(ROOT / 'scripts/edival_scoring.py'), 'prepare', '--output', str(output),
                       '--inference', str(inference_root), '--num-shards', '4', '--setup', str(a.base / 'setup/edival')]
        else:
            command = [str(ROOT / 'scripts/score_mice_qwen_once.py'), 'prepare', '--output', str(output),
                       '--inference', str(inference_root), '--allow-full']
        if output.exists() and not (output / 'preflight.json').exists():
            output.rename(output.with_name(output.name + '_interrupted_prepare_' + str(time.time_ns())))
        if not (output / 'preflight.json').exists():
            execute(command, a.gpu, 'EdiVal' if run.startswith('edival') else 'lance', a.base / 'logs' / (run + '_prepare.log'), temporary)
        if read(output / 'preflight.json')['status'] != 'passed':
            raise ValueError('Scoring preflight failed')
    elif job['kind'] == 'score':
        stage = job['stage']
        if run.startswith('edival'):
            command = [str(ROOT / 'scripts/edival_scoring.py'), 'worker', '--output', str(output),
                       '--stage', stage, '--shard', str(job['shard']), '--gpu', a.gpu]
            env = ENVS['metrics_edival' if stage == 'metrics' else stage]
        else:
            command = [str(ROOT / 'scripts/sixrun_score_worker.py'), '--output', str(output),
                       '--inference', str(inference_root), '--stage', stage, '--shard', str(job['shard'])]
            env = ENVS['metrics_mice' if stage == 'metrics' else stage]
        execute(command, a.gpu, env, output / f"{stage}_{job['shard']}.log", temporary)
    else:
        if run.startswith('edival'):
            commands = [[str(ROOT / 'scripts/edival_scoring.py'), 'aggregate', '--output', str(output)]]
        else:
            commands = [[str(ROOT / 'scripts/score_mice_qwen_once.py'), stage, '--output', str(output),
                         '--inference', str(inference_root), '--allow-full'] for stage in ['report', 'validate']]
        for command in commands:
            execute(command, a.gpu, 'EdiVal' if run.startswith('edival') else 'lance', output / 'aggregate.log', temporary)
        write(output / 'completion.json', {'state': 'completed', 'exit_code': 0, 'finished_at': now()})


def advance(base, jobs):
    """Create dependency tasks once under the queue lock, without holding a GPU."""
    plan = read(base / 'plan.json')
    ids = {j['id'] for j in jobs}
    for run in plan['runs']:
        infer = [j for j in jobs if j['kind'] == 'infer' and j['run'] == run]
        if not all(j['state'] == 'completed' for j in infer):
            continue
        audit_id = f'audit:{run}'
        if audit_id not in ids:
            jobs.append({'id': audit_id, 'kind': 'audit', 'run': run, 'state': 'pending'})
        audit = next(j for j in jobs if j['id'] == audit_id)
        if audit['state'] != 'completed' or run.startswith('imgedit'):
            continue
        prep_id = f'prepare_score:{run}'
        if prep_id not in ids:
            jobs.append({'id': prep_id, 'kind': 'prepare_score', 'run': run, 'state': 'pending'})
        prep = next(j for j in jobs if j['id'] == prep_id)
        if prep['state'] != 'completed':
            continue
        stages = ['if', 'metrics', 'hps'] if run.startswith('edival') else ['metrics', 'judge']
        for stage in stages:
            # Only MICE judge requires its same-session metrics shard first.
            for shard in range(4):
                jid = f'score:{run}:{stage}:{shard}'
                if jid not in ids:
                    jobs.append({'id': jid, 'kind': 'score', 'run': run, 'stage': stage,
                                 'shard': shard, 'state': 'pending'})
        relevant = [j for j in jobs if j['kind'] == 'score' and j['run'] == run]
        jid = f'aggregate:{run}'
        if relevant and all(j['state'] == 'completed' for j in relevant) and jid not in ids:
            jobs.append({'id': jid, 'kind': 'aggregate', 'run': run, 'state': 'pending'})


def claim(a):
    with queue_lock(a.base):
        jobs = read(a.base / 'queue.json')
        advance(a.base, jobs)
        ready = []
        inference_remaining = any(j['kind'] == 'infer' and j['state'] != 'completed' for j in jobs)
        active_scores = sum(j['kind'] == 'score' and j['state'] == 'running' for j in jobs)
        for job in jobs:
            if job['state'] not in ['pending', 'failed'] or job.get('attempts', 0) >= 3:
                continue
            if job['kind'] == 'score' and inference_remaining and active_scores >= 2:
                continue
            if job.get('retry_after', 0) > time.time():
                continue
            if job['kind'] == 'infer' and job.get('owner_host') not in [None, a.host]:
                continue
            if job['kind'] == 'score' and job['run'].startswith('mice') and job['stage'] == 'judge':
                dependency = f"score:{job['run']}:metrics:{job['shard']}"
                if next(j for j in jobs if j['id'] == dependency)['state'] != 'completed':
                    continue
            ready.append(job)
        # EdiVal scoring is on the critical path; overlap it with remaining inference.
        order = {'audit': 0, 'prepare_score': 1, 'aggregate': 0, 'score': 2, 'infer': 3}
        ready.sort(key=lambda j: (order[j['kind']], 0 if j['run'].startswith('edival') else 1,
                                  0 if j.get('stage') in ['judge', 'if'] else 1))
        job = ready[0] if ready else None
        if job:
            job.update(state='running', owner_host=a.host, owner_gpu=a.gpu, owner_pid=os.getpid(),
                       started_at=now(), attempts=job.get('attempts', 0)+1)
        write(a.base / 'queue.json', jobs)
        return dict(job) if job else None


def finish(a, job, error=None):
    with queue_lock(a.base):
        jobs = read(a.base / 'queue.json')
        current = next(j for j in jobs if j['id'] == job['id'])
        if current['owner_pid'] != os.getpid() or current['owner_host'] != a.host:
            raise ValueError('Job ownership changed')
        current.update(state='failed' if error else 'completed', finished_at=now())
        if error:
            current.setdefault('errors', []).append({'at': now(), 'message': error})
            current['retry_after'] = time.time()+30
        write(a.base / 'queue.json', jobs)


def worker(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    for name, pin in read(a.base / 'plan.json')['source_hashes'].items():
        if sha(ROOT / name) != pin:
            raise ValueError('Frozen runtime changed: '+name)
    samples = {(bench, s.session_id): s for bench in ['edival', 'mice', 'imgedit']
               for s in load_samples(DEFAULT_ROOTS[bench], 'all', bench)}
    temporary = tempfile.mkdtemp(prefix=f'umm-sixrun-{a.host}-{a.gpu}.', dir='/tmp')
    os.environ.update(TMPDIR=temporary, XDG_CACHE_HOME=temporary+'/cache', VLLM_CACHE_ROOT=temporary+'/vllm',
                      TORCHINDUCTOR_CACHE_DIR=temporary+'/inductor', TRITON_CACHE_DIR=temporary+'/triton',
                      HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    backend = None
    try:
        while True:
            job = claim(a)
            status = {'host': a.host, 'gpu': a.gpu, 'pid': os.getpid(), 'updated_at': now(),
                      'job': job, 'state': 'working' if job else 'waiting', 'temporary': temporary}
            write(a.base / 'workers' / f'{a.host}_{a.gpu}.json', status)
            if not job:
                with queue_lock(a.base):
                    jobs = read(a.base / 'queue.json')
                    done = all(j['state'] == 'completed' for j in jobs)
                if backend is not None:
                    backend.close()
                    backend = None
                if done:
                    write(a.base / 'completion.json', {'state': 'completed', 'finished_at': now(), 'expected_turns': 7928})
                    return
                time.sleep(10)
                continue
            try:
                if job['kind'] == 'infer':
                    if backend is None:
                        from lance_mice.backend import OmniBackend
                        backend = OmniBackend(Path('/home/chs/model/Lance'), Settings(), audit=True)
                    inference(a, job, backend, samples)
                else:
                    if backend is not None:
                        backend.close()
                        backend = None
                        time.sleep(2)
                    if job['kind'] == 'audit':
                        finalize_inference(a.base, job['run'])
                    else:
                        score(a, job, temporary)
                finish(a, job)
                print(now(), 'COMPLETED', job['id'], flush=True)
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                finish(a, job, error)
                # Destroy the potentially invalid CUDA engine; the external supervisor restarts.
                raise
    finally:
        if backend is not None:
            backend.close()
        shutil.rmtree(temporary)


def status(a):
    jobs = read(a.base / 'queue.json')
    runs = {}
    for run, spec in read(a.base / 'plan.json')['runs'].items():
        infer = [j for j in jobs if j['run'] == run and j['kind'] == 'infer']
        runs[run] = {**spec, 'completed_sessions': sum(j['state'] == 'completed' for j in infer),
                     'completed_turns': sum(j['turns'] for j in infer if j['state'] == 'completed'),
                     'states': dict(Counter(j['state'] for j in jobs if j['run'] == run))}
    print(json.dumps({'at': now(), 'runs': runs, 'states': dict(Counter(j['state'] for j in jobs)),
                      'failures': [j for j in jobs if j['state'] == 'failed'],
                      'workers': [read(p) for p in (a.base / 'workers').glob('*.json')]}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'worker', 'status'])
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--host', choices=['a800_0', 'a800_1'])
    p.add_argument('--gpu', choices=['0', '1'])
    p.add_argument('--local', type=Path, default=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/sixrun_20261007'))
    a = p.parse_args()
    if a.action == 'worker' and (a.host is None or a.gpu is None):
        p.error('worker requires host and GPU')
    {'init': initialize, 'worker': worker, 'status': status}[a.action](a)

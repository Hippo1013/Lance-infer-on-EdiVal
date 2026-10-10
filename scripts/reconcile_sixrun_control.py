#!/usr/bin/env python3
"""Reconcile first committed receipts after the GPFS host-local flock failure."""
import argparse
from pathlib import Path
from sixrun_queue import ROOT, read, write, sha, now


def inventory(a):
    local = Path(read(a.base/'plan.json')['local_outputs'])
    rows = {}
    for path in local.glob('*/run/*/*/session.json'):
        run = path.relative_to(local).parts[0]
        spec = read(path)
        rows[f"infer:{run}:{spec['session_id']}"] = {
            'host': a.host, 'session_path': str(path), 'mtime_ns': path.stat().st_mtime_ns,
            'complete_turns': sum((path.parent/f'turn_{t}.json').exists() for t in range(1,4))}
    write(a.base/'setup'/f'inventory_{a.host}.json', rows)


def reconcile(a):
    plan = read(a.base/'plan.json')
    old_root = Path(plan['runtime'])
    old_pins = plan['source_hashes']
    new_pins = {name: sha(ROOT/name) for name in old_pins}
    changed = {name for name in old_pins if old_pins[name] != new_pins[name]}
    if changed != {'scripts/sixrun_queue.py', 'scripts/sixrun_score_worker.py'}:
        raise ValueError('Changes beyond the two concurrency controllers: '+str(changed))
    for name, pin in old_pins.items():
        if sha(old_root/name) != pin:
            raise ValueError('Original frozen runtime changed: '+name)
    evidence = read(a.base/'setup/atomic_lock_stress.json')
    if evidence['count'] != 160 or len(evidence['workers']) != 4 or set(evidence['workers'].values()) != {40}:
        raise ValueError('Four-process, two-host atomic-lock stress test did not pass')
    inventories = {host: read(a.base/'setup'/f'inventory_{host}.json') for host in ['a800_0','a800_1']}
    queue = read(a.base/'queue.json')
    original_queue = queue.copy()
    completed, retained_duplicates = 0, []
    for job in queue:
        root = a.base/'inference'/job['run']
        receipt_path = root/'receipts'/(job['session'].replace('/','__')+'.json')
        candidates = [v[job['id']] for v in inventories.values() if job['id'] in v]
        if len(candidates) > 1:
            retained_duplicates.append({'id': job['id'], 'candidates': candidates,
                                        'policy': 'first canonical committed receipt, otherwise canonical source owner; no quality selection'})
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['status'] != 'passed' or receipt['fingerprint'] != plan['runs'][job['run']]['fingerprint']:
                raise ValueError('Invalid committed receipt')
            for name,pin in receipt['image_metadata_hashes'].items():
                if sha(root/'run'/job['session']/name) != pin:
                    raise ValueError('Canonical first-commit evidence changed')
            job.update(state='completed', owner_host=receipt['host'], owner_gpu=receipt['gpu'],
                       finished_at=receipt['at'], reconciliation='verified first canonical receipt')
            completed += 1
        else:
            canonical_spec = root/'run'/job['session']/'session.json'
            if canonical_spec.exists():
                source = read(canonical_spec)['source']
                host = 'a800_1' if source.startswith('/media/damoxing/') else 'a800_0'
                if not any(r['host'] == host for r in candidates):
                    raise ValueError('Canonical source has no producer inventory')
            elif candidates:
                host = min(candidates,key=lambda r:r['mtime_ns'])['host']
            else:
                host = None
            job.update(state='pending', owner_host=host, attempts=0,
                       reconciliation='resume original local transaction; retained every prior attempt')
            job.pop('retry_after',None)
    write(a.base/'setup/control_migration_v2_v3.json', {
        'status':'passed', 'at':now(), 'previous_runtime':str(old_root), 'runtime':str(ROOT),
        'changed_existing_files':sorted(changed), 'original_source_hashes':old_pins,
        'new_source_hashes':new_pins, 'committed_sessions_restored':completed,
        'duplicate_attempts_retained':retained_duplicates,
        'cause':'GPFS flock independently granted on both hosts; atomic mkdir cross-host stress count 160/160',
        'checks':['all generator/protocol/attention/scoring criteria files identical',
                  'original frozen runtime preserved', 'all first canonical receipt artifact hashes verified'],
        'reconciliation_source_sha256':sha(Path(__file__))})
    plan.update(runtime=str(ROOT),source_hashes=new_pins,control_revision='v3 atomic cross-host directory lease')
    write(a.base/'plan.json',plan)
    for path in (a.base/'inference').glob('*/source_hashes.json'):
        write(path,new_pins)
    write(a.base/'queue.json',queue)
    print('RECONCILIATION_PASSED',completed,'duplicate attempts',len(retained_duplicates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inventory','reconcile'])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--host',choices=['a800_0','a800_1'])
    a=p.parse_args()
    {'inventory':inventory,'reconcile':reconcile}[a.action](a)

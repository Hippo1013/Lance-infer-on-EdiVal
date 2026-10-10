#!/usr/bin/env python3
"""Host-sharded execution for the accepted EdiVal v2 observations.

prepare must finish before compute; scientific inputs are never overwritten.
Run in the existing numerical analysis environment. All outputs remain on servers.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
from core import analyze_turn, compact, paired_turns, split_clusters

TASK='edival_extended_20261010_v1'
PROJECT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
ACCEPTED=PROJECT/'outputs/attention_analysis/edival_semantic_20261009_v2'
ROOT=PROJECT/'outputs/attention_analysis'/TASK
HOSTS={'a800_0':'aibox-r61097f954fe-7d74d99d65-2zzkn','a800_1':'aibox-r7df77faf487-f8c468557-b9c9s'}


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_name(path.name+'.part')
    q.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    os.replace(q,path)


def save_array(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    q=path.with_name(path.name+'.part')
    with q.open('wb') as f:np.savez_compressed(f,**value)
    os.replace(q,path)


def prepare():
    if platform.node()!=HOSTS['a800_0']:raise ValueError('Prepare only on source archive host')
    source=ACCEPTED/'metadata/source_manifest.jsonl.gz'
    with gzip.open(source,'rt') as f:items=[json.loads(x) for x in f]
    if len(items)!=1716 or len({x['session_id'] for x in items})!=572:raise ValueError('Incomplete accepted inputs')
    if {x['attention_version'] for x in items}!={'target-token-region-stats-v2'}:raise ValueError('Mixed observation versions')
    accept=json.loads((ACCEPTED/'local_acceptance.json').read_text())
    # Keep complete existing acceptance record on the server as provenance.
    if 'passed' not in json.dumps(accept):raise ValueError('Missing accepted scientific version')
    metadata=ROOT/'metadata';metadata.mkdir(parents=True,exist_ok=True)
    input_path=metadata/'source_manifest.jsonl.gz'
    if input_path.exists() and sha(input_path)!=sha(source):raise ValueError('Conflicting existing task')
    if not input_path.exists():
        import shutil
        shutil.copy2(source,input_path)
    split=split_clusters(items)
    definition={
        'task_id':TASK,'source_task':'edival_semantic_20261009_v2','source_manifest_sha256':sha(source),
        'source_acceptance_sha256':sha(ACCEPTED/'local_acceptance.json'),
        'created_at':now(),'sessions':572,'turns':1716,'clusters':570,
        'split':split,'split_method':'SHA256 seeded ranking of unique original RGB hashes; first 285 exploration, rest confirmation',
        'low_mass_threshold':1e-6,'bootstrap_reps':2000,'bootstrap_seed':20261010,
        'near_zero_change_thresholds':{'mass':1e-4,'conditional_share':.01,'entropy':.01,'TV':.01},
        'sensitivity_trim':.05,'top_overlap_ties':'stable token/region index; report TV alongside overlap',
        'candidate_selection':'exploration only; fixed contiguous 3-step x 4-layer windows; confirmation read after frozen candidate file',
        'scope':'all EdiVal sessions; no scores, inference, model intervention or other benchmarks',
        'observer_boundary':'query-averaged key distributions; only original group Query mean/std/quantiles',
    }
    contract=metadata/'design.json'
    if contract.exists():
        prior=json.loads(contract.read_text())
        if prior['source_manifest_sha256']!=definition['source_manifest_sha256'] or prior['split']!=definition['split']:raise ValueError('Incompatible existing design')
    else:atomic_json(contract,definition)
    return definition


def worker(args):
    sid,items=args
    from layout import layout_audit
    turns={};records=[];identity={}
    for item in sorted(items,key=lambda x:x['generation_turn']):
        t=item['generation_turn'];p=Path(item['original_path'])
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError(('Source hash',item['source_id']))
        jp=p.with_name(f'turn_{t}.json');j=json.loads(jp.read_text())
        if j['input_hashes']!=item['input_hashes'] or j['output_hash']!=item['output_hash']:raise ValueError('History identity mismatch')
        for im,h in enumerate(j['input_hashes']):
            if im in identity and identity[im]!=h:raise ValueError('Same object image identity changed')
            identity[im]=h
        with np.load(p,allow_pickle=False) as z:
            layout_audit(z,j)
            result=analyze_turn(z)
            axes={'timesteps':z['timesteps'].copy(),'layer_ids':z['layer_ids'].copy(),'source_sha256':np.array(item['sha256'])}
        turns[t]=result
        output=ROOT/'arrays'/sid/f'turn_{t}.npz'
        save_array(output,dict(compact(*result[:3]),**axes))
        token_metadata={}
        for g,meta in result[4].items():
            tr=next(x for x in j['backend']['positive_trace'] if x.get('role') in ['current','history'] and x['turn']==int(g[1:]))
            token_metadata[g]=dict(meta,text=tr['text'],pieces=[tr['text'][a:b] for a,b in meta['offsets']])
        atomic_json(ROOT/'objects'/sid/f'turn_{t}.json',dict(session_id=sid,turn=t,image_hashes=j['input_hashes'],
            tokens=token_metadata,geometry=[x for x in j['backend']['positive_trace'] if x['kind'] in ['vit','vae']],source_sha256=item['sha256']))
        records.append({'source_id':item['source_id'],'source_sha256':item['sha256'],'output':str(output.relative_to(ROOT)),'output_sha256':sha(output)})
    if set(turns)!={1,2,3}:raise ValueError('Missing turn')
    if any(not np.array_equal(np.load(ROOT/'arrays'/sid/f'turn_1.npz')['timesteps'],np.load(ROOT/'arrays'/sid/f'turn_{t}.npz')['timesteps']) for t in [2,3]):raise ValueError('Misaligned timesteps')
    pairs=paired_turns(turns)
    for key,data in pairs.items():save_array(ROOT/'paired'/sid/(key+'.npz'),data)
    receipt={'session_id':sid,'status':'passed','records':records,'paired_objects':len(pairs),'finished_at':now(),'identity_checked':True}
    atomic_json(ROOT/'receipts'/(sid.replace('/','__')+'.json'),receipt)
    return receipt


def compute(host,workers):
    if platform.node()!=HOSTS[host]:raise ValueError('Host mismatch')
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:items=[json.loads(x) for x in f]
    grouped=defaultdict(list)
    for x in items:
        if x['host']==host:grouped[x['session_id']].append(x)
    expected=279 if host=='a800_0' else 293
    if len(grouped)!=expected:raise ValueError('Unexpected host shard')
    done=[];start=now()
    status_path=ROOT/f'{host}_status.json'
    atomic_json(status_path,dict(task_id=TASK,stage='computing',started_at=start,sessions_done=0,sessions_total=expected))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(worker,(sid,grouped[sid])):sid for sid in sorted(grouped)}
            for future in as_completed(pending):
                done.append(future.result())
                atomic_json(status_path,dict(task_id=TASK,stage='computing',started_at=start,at=now(),sessions_done=len(done),sessions_total=expected))
        files=[]
        for folder in ['arrays','paired','objects','receipts']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
        atomic_json(ROOT/f'{host}_manifest.json',dict(task_id=TASK,host=host,files=files))
        atomic_json(status_path,dict(task_id=TASK,stage='computed',sessions_done=len(done),turns_done=len(done)*3,started_at=start,finished_at=now(),
            manifest_sha256=sha(ROOT/f'{host}_manifest.json'),python=platform.python_version(),numpy=np.__version__,
            code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    except Exception as exc:
        atomic_json(status_path,dict(task_id=TASK,stage='failed',at=now(),sessions_done=len(done),error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compute']);p.add_argument('--host',choices=list(HOSTS));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:compute(a.host,a.workers)

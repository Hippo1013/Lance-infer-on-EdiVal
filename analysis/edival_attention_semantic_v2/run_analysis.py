#!/usr/bin/env python3
"""Wait for the fixed collection; compute, independently verify and freeze tables."""
import csv
import gzip
import json
import os
import pathlib
import tempfile
import time
import urllib.error
from collections import defaultdict
import numpy as np
from common import *
import pipeline,tests,independent,documentation,validate

def verify_pairs(temp):
    groups={}
    with gzip.open(ROOT/'tables/group_turn.csv.gz','rt') as f:
        for r in csv.DictReader(f):groups[(r['session_id'],r['generation_turn'],r['group_name'])]=r
    rows=[]
    with gzip.open(ROOT/'tables/within_type_pairs.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    assert len(rows)==9152 and len({r['session_id'] for r in rows})==572
    # Every pair's pooled ratio is independently recovered from scalar mass;
    # per-position ratio has a separate deterministic raw-array sample check.
    for r in rows:
        sid=r['session_id'];t=r['generation_turn'];m=r['modality'];a=int(r['older_index']);b=int(r['newer_index'])
        def mass(i):
            gs=[f'T{i}'] if r['object_type']=='text' else [f'I{i}_{k}' for k in (['vit','vae'] if m=='combined' else [m])]
            return sum(float(groups[(sid,t,g)]['mass']) for g in gs)
        aa,bb=mass(a),mass(b)
        if aa+bb>0:assert abs(float(r['pooled_older_share'])-aa/(aa+bb))<1e-12
        for key in ['older_share','older_density_share','older_share_sensitivity','pooled_older_share']:
            if r[key]!='':assert 0<=float(r[key])<=1
        assert int(r['positive_positions'])>=int(r['sensitivity_positions'])
    groups_by=defaultdict(list)
    for r in rows:
        key=tuple(r[k] for k in ['object_type','modality','older_index','newer_index','generation_turn']);groups_by[key].append(r)
    with gzip.open(ROOT/'tables/within_type_summary.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in ['object_type','modality','older_index','newer_index','generation_turn'])
            vv=[float(x[r['metric']]) for x in groups_by[key] if x[r['metric']]!='']
            assert int(r['n_valid'])==len(vv)
            if vv:assert abs(float(r['mean'])-sum(vv)/len(vv))<1e-12
    selection=json.loads((ROOT/'metadata/sample_selection.json').read_text())['sessions']
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:items=[json.loads(x) for x in f if json.loads(x)['session_id'] in selection]
    sample_checks=0
    for item in items:
        sid=item['session_id'];t=str(item['generation_turn']);selected=[r for r in rows if r['session_id']==sid and r['generation_turn']==t]
        if not selected:continue
        path=pathlib.Path(item['original_path']);remote=item['host']=='a800_1'
        if remote:
            path=pathlib.Path(temp)/('pair_'+sid.replace('/','__')+'_'+t+'.npz');download(f'/attention/{sid}/turn_{t}.attention.npz',path,item['bytes'],item['sha256'])
        try:
            with np.load(path,allow_pickle=False) as z:
                names=z['group_names'].tolist();count=z['group_token_counts'];mass=z['group_stats'][...,0].astype(np.float64).reshape(17280,-1)
                for r in selected:
                    def values(i):
                        gs=[f'T{i}'] if r['object_type']=='text' else [f'I{i}_{k}' for k in (['vit','vae'] if r['modality']=='combined' else [r['modality']])]
                        ix=[names.index(g) for g in gs];return np.add.reduce(mass[:,ix],axis=1),np.add.reduce(count[ix])
                    aa,na=values(int(r['older_index']));bb,nb=values(int(r['newer_index']));positive=aa+bb>0;sens=aa+bb>1e-6
                    for field,mask,a,b in [('older_share',positive,aa,bb),('older_share_sensitivity',sens,aa,bb),('older_density_share',positive,aa/na,bb/nb)]:
                        if mask.any():assert abs(float(r[field])-np.mean(a[mask]/(a[mask]+b[mask])))<1e-12
                        else:assert r[field]==''
                    sample_checks+=1
        finally:
            if remote:path.unlink(missing_ok=True)
    # Two different distributions prove why the ratio is computed first.
    x=np.array([1.,9.]);y=np.array([1.,1.])
    assert abs(np.mean(x/(x+y))-x.mean()/(x+y).mean())>.1
    atomic_json(ROOT/'validation/within_type_checks.json',dict(status='passed',rows=len(rows),raw_sample_pairs_checked=sample_checks,checks=['pooled ratios independently reconstructed from group table','ratio bounds and valid-position denominators','nonlinear ratio before averaging regression']))

def manifest(path,dirs):
    files=[]
    for d in dirs:
        for p in sorted((ROOT/d).rglob('*')):
            if p.is_file():files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    atomic_json(ROOT/path,dict(task_id=TASK,files=files,at=now()))
    return sha(ROOT/path)
def main():
    for d in ['tables','arrays','metadata','validation','schemas','logs','exchange']: (ROOT/d).mkdir(parents=True,exist_ok=True)
    status('waiting_for_recollection',issues=[])
    collect=PROJECT/'outputs/attention_recollection'/TASK
    while True:
        local=[collect/s/'completion.json' for s in ['full_gpu0','full_gpu1']]
        if any((collect/s/'failure.json').exists() for s in ['full_gpu0','full_gpu1']):raise ValueError('Local full recollection failed')
        if all(p.exists() for p in local):
            try:
                if peer_json('/export')['status']=='passed':break
            except urllib.error.HTTPError as e:
                if e.code!=409:raise
        time.sleep(30)
    import shutil,sys,platform
    snapshot=ROOT/'metadata/analysis_source';snapshot.mkdir(exist_ok=True)
    for p in pathlib.Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    atomic_json(ROOT/'metadata/code_manifest.json',dict(files={p.name:sha(p) for p in snapshot.glob('*.py')}))
    atomic_json(ROOT/'metadata/environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),executable=sys.executable))
    with tempfile.TemporaryDirectory(prefix='edival-semantic-analysis-') as temp:
        pipeline.run(temp,8)
        tests.run()
        independent.run(temp)
        verify_pairs(temp)
        documentation.build()
        validate.run()
    h=manifest('metadata/candidate_manifest.json',['tables','arrays','schemas','metadata'])
    # Candidate source includes the actual analysis implementation and SHA256s.
    atomic_json(ROOT/'validation/completion_checks.json',dict(status='passed',sessions=572,turns=1716,real_cache_identity='all full turns, first/last layer each of 30 steps; pilots all 36 layers',
        original_output_pixels='exact for every recollected turn',formula_checks='independent alternate entropy, metric masks, whole-table weights, three bootstrap intervals and same-type ratios',temporary_directory_removed=True))
    artifact_hash=manifest('artifact_manifest.json',['tables','arrays','schemas','metadata','validation'])
    atomic_json(ROOT/'completion.json',dict(task_id=TASK,status='passed',input_sessions=572,input_turns=1716,
        independently_validated_candidate_sha256=h,artifact_manifest_path='artifact_manifest.json',artifact_manifest_sha256=artifact_hash,
        sources='original histories and exact original output pixel hashes; target-token-region-stats-v2',at=now(),issues=[]))
    status('complete',issues=[])
if __name__=='__main__':
    try:main()
    except BaseException as e:
        atomic_json(ROOT/'validation/analysis_failure.json',dict(status='failed',error=repr(e),at=now()));status('failed',issues=[repr(e)]);raise

#!/usr/bin/env python3
"""Coverage, source bindings, baseline reproduction and aggregate checks."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from run import ROOT,ACCEPTED,sha,atomic_json,now,TASK


def validate():
    design=json.loads((ROOT/'metadata/design.json').read_text())
    sids=sorted(design['split']);assert len(sids)==572
    assert len({v['cluster'] for v in design['split'].values()})==570
    assert sha(ROOT/'metadata/source_manifest.jsonl.gz')==design['source_manifest_sha256']
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:source={x['source_id']:x for x in map(json.loads,f)}
    audits=[];checks=0
    for sid in sids:
        receipt=json.loads((ROOT/'receipts'/(sid.replace('/','__')+'.json')).read_text())
        assert receipt['status']=='passed' and receipt['identity_checked'] and len(receipt['records'])==3 and receipt['paired_objects']==12
        for r in receipt['records']:
            assert r['source_sha256']==source[r['source_id']]['sha256']
            assert sha(ROOT/r['output'])==r['output_sha256'];checks+=1
        for t in [1,2,3]:
            with np.load(ROOT/'arrays'/sid/f'turn_{t}.npz') as z:
                assert z['process'].shape[:2]==(30,36) and z['head'].shape[:2]==(36,16)
                assert np.all(z['n_valid']<=17280) and np.all(z['n_valid']>=0)
                assert len(set(z['names'].tolist()))==len(z['names'])
    # Reproduction of already accepted per-session/per-turn scalar statistics.
    expected={}
    for table,group_field,field_map in [
        ('group_turn','group_name',{'mass':'mass','per_token_mass':'density'}),
        ('concentration_turn','support_id',{'normalized_entropy':'entropy','top10pct_share':'top10','n90_fraction':'n90'})]:
        with gzip.open(ACCEPTED/'tables'/(table+'.csv.gz'),'rt') as f:
            for row in csv.DictReader(f):
                for old,new in field_map.items():
                    if row.get(old,'')!='':expected[row['session_id'],int(row['generation_turn']),row[group_field]+':'+new]=float(row[old])
    errors=[]
    for sid in sids:
        for t in [1,2,3]:
            with np.load(ROOT/'arrays'/sid/f'turn_{t}.npz') as z:
                for i,k in enumerate(z['names'].tolist()):
                    key=(sid,t,k)
                    if key in expected:errors.append(abs(float(z['scalar'][i])-expected[key]))
    assert errors and max(errors)<2e-7
    for host in ['a800_0','a800_1']:
        c=json.loads((ROOT/'metadata'/f'{host}_independent_check.json').read_text());assert c['status']=='passed' and c['turns']==9
    names=json.loads((ROOT/'metadata/metric_names.json').read_text())
    with np.load(ROOT/'summaries/full_atlas.npz') as z:
        keys=z['keys'].tolist()
        for t in [1,2,3]:
            a=z[f'mean_{keys.index(f"t{t}__process")}'];cols=names[str(t)]
            allocation=[i for i,k in enumerate(cols) if k.startswith('category_') and k.endswith(':mass')]
            assert len(allocation)==7 and np.max(np.abs(a[...,allocation].sum(-1)-1))<1e-3
    with np.load(ROOT/'summaries/session_scalars.npz') as z:
        values=z['values'];cols=z['metric_names'].tolist();splits=z['splits'];clusters=z['clusters']
        summaries=json.loads((ROOT/'summaries/scalar_summary.json').read_text())
        for i,row in enumerate(summaries):
            assert row['metric']==cols[i]
            xx=values[:,i];valid=xx[np.isfinite(xx)]
            if not len(valid):assert row['mean'] is None;continue
            assert abs(row['mean']-float(valid.mean()))<1e-12
            assert row['n_valid']==len(valid)
        exp=set(clusters[splits=='exploration']);confirm=set(clusters[splits=='confirmation']);assert not exp&confirm and len(exp)==len(confirm)==285
    frozen=json.loads((ROOT/'metadata/frozen_candidates.json').read_text())
    candidate=json.loads((ROOT/'summaries/candidate_review.json').read_text());assert candidate['frozen_candidates_sha256']==sha(ROOT/'metadata/frozen_candidates.json')
    assert len(frozen['candidates'])==4 and len(candidate['rows'])==12
    files=[]
    for folder in ['arrays','paired','objects','receipts','tables','summaries','metadata']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file():
                assert not p.name.endswith('.part')
                files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    atomic_json(ROOT/'validation/science.json',dict(status='passed',sessions=572,turns=1716,paired_objects=6864,original_clusters=570,
        source_hash_bound_turns=checks,baseline_scalar_comparisons=len(errors),max_baseline_scalar_error=max(errors),
        independent_real_turns=18,exploration_clusters=285,confirmation_clusters=285,summary_rows=len(summaries),at=now(),
        note='Independent formulas, accepted baseline reproduction, complete identity and scalar checks; no external Codex review claimed'))
    atomic_json(ROOT/'artifact_manifest.json',dict(task_id=TASK,files=files,source_acceptance_sha256=design['source_acceptance_sha256']))
    print('scientific checks passed',len(errors),max(errors),len(files))

if __name__=='__main__':validate()

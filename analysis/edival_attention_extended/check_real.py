#!/usr/bin/env python3
"""Independent formula checks against real input arrays, not the core functions."""
from collections import defaultdict
import argparse
import gzip
import json
import math
from pathlib import Path
import platform
import numpy as np
from run import ROOT, HOSTS, atomic_json, sha, now


def main(host):
    assert platform.node()==HOSTS[host]
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:items=[json.loads(x) for x in f if json.loads(x)['host']==host]
    sids=sorted({x['session_id'] for x in items});selected={sids[0],sids[len(sids)//2],sids[-1]}
    checks=[]
    for item in items:
        sid=item['session_id'];t=item['generation_turn']
        if sid not in selected:continue
        with np.load(item['original_path'],allow_pickle=False) as z, np.load(ROOT/'arrays'/sid/f'turn_{t}.npz',allow_pickle=False) as out:
            names=z['group_names'].tolist();stats=z['stat_names'].tolist();cols=out['names'].tolist();m=z['group_stats'][...,stats.index('mean')].astype(float);sd=z['group_stats'][...,stats.index('std')].astype(float)
            errors=[];count=0
            for i,g in enumerate(names):
                r=np.full(m.shape[:3],np.nan);valid=m[...,i]**2+sd[...,i]**2>0
                r[valid]=m[...,i][valid]**2/(m[...,i][valid]**2+sd[...,i][valid]**2)
                for key,a in [(g+':mass',m[...,i]),(g+':query_R',r)]:
                    scalar=float(np.nanmean(a));saved=out['scalar'][cols.index(key)]
                    if not (np.isnan(scalar) and np.isnan(saved)):errors.append(abs(scalar-saved))
                    sums=np.nansum(a,axis=2);n=np.isfinite(a).sum(2);expected=np.divide(sums,n,out=np.full((30,36),np.nan),where=n>0)
                    actual=out['process'][...,cols.index(key)]
                    assert np.array_equal(np.isfinite(expected),np.isfinite(actual))
                    errors.extend(np.abs(expected-actual)[np.isfinite(expected)].tolist());count+=1081
                if g.startswith(('T','I')):
                    # These profiles use the corresponding fine-support sums,
                    # not the separately saved group SDPA channel. Half-precision
                    # observation channels need not agree bit-for-bit, especially
                    # after conditioning on very small probability masses.
                    if g.startswith('T'):
                        fine=z['text_mean'][...,z['text_token_turns']==int(g[1:])].astype(float).sum(-1)
                    else:
                        index=int(g.split('_')[0][1:]);mod=int(g.endswith('vae'))
                        fine=z['region_mean'][...,index,mod,:,:].astype(float).sum((-1,-2))
                    head_total=fine.sum(axis=2)
                    ranked=np.sort(fine,axis=2)[...,::-1]
                    for k in [1,4]:
                        expected=np.divide(ranked[...,:k].sum(2),head_total,out=np.full((30,36),np.nan),where=head_total>0)
                        profile=out['profile_'+str(out['profile_names'].tolist().index(g+f':top{k}_head_share'))]
                        errors.extend(np.abs(expected-profile)[np.isfinite(expected)].tolist());count+=int(np.isfinite(expected).sum())
            # Independent scalar entropy/top share at each aligned observation.
            for instruction in range(1,t+1):
                x=z['text_mean'][...,z['text_token_turns']==instruction].astype(float).reshape(-1,int((z['text_token_turns']==instruction).sum()))
                h=[];top=[]
                for row in x:
                    total=float(sum(row))
                    if total<=0:continue
                    p=[float(v)/total for v in row]
                    if len(p)>1:h.append(-sum(v*math.log(v) for v in p if v>0)/math.log(len(p)))
                    top.append(sum(sorted(p,reverse=True)[:math.ceil(.1*len(p))]))
                for key,vals in [(f'T{instruction}:entropy',h),(f'T{instruction}:top10',top)]:
                    actual=out['scalar'][cols.index(key)]
                    if vals:errors.append(abs(float(np.mean(vals))-actual));count+=len(vals)
            # Independent same-round ratios, before averaging observations.
            if t==3:
                for ga,gb in [('T1','T2'),('I1_vit','I2_vit')]:
                    a=m[...,names.index(ga)];b=m[...,names.index(gb)];den=a+b;valid=den>0
                    ratios=a[valid]/den[valid]
                    expected=float(ratios.mean());key=f'pair_{ga}_{gb}:older_share'
                    errors.append(abs(expected-out['scalar'][cols.index(key)]))
            maximum=max(errors)
            if maximum>2e-7:raise ValueError(('Independent formula mismatch',item['source_id'],maximum))
            checks.append(dict(source_id=item['source_id'],comparisons=count,max_absolute_error=maximum,source_sha256=sha(item['original_path'])))
    # Paired differences are rederived directly from the two original arrays.
    pair_checks=[]
    for sid in sorted(selected):
        a=next(x for x in items if x['session_id']==sid and x['generation_turn']==1)
        b=next(x for x in items if x['session_id']==sid and x['generation_turn']==3)
        with np.load(a['original_path']) as za,np.load(b['original_path']) as zb:
            for g in ['T1','I0_vae']:
                if g=='T1':
                    x=za['text_mean'][...,za['text_token_turns']==1].astype(float)
                    y=zb['text_mean'][...,zb['text_token_turns']==1].astype(float)
                else:
                    x=za['region_mean'][...,0,1,:,:].astype(float).reshape(30,36,16,64)
                    y=zb['region_mean'][...,0,1,:,:].astype(float).reshape(30,36,16,64)
                ma=x.sum(-1);mb=y.sum(-1);valid=(ma>0)&(mb>0)
                px=np.divide(x,ma[...,None],out=np.zeros_like(x),where=ma[...,None]>0)
                py=np.divide(y,mb[...,None],out=np.zeros_like(y),where=mb[...,None]>0)
                dist=np.abs(px-py).sum(-1)/2
                with np.load(ROOT/'paired'/sid/f'1_3__{g}.npz') as out:
                    pos=out['names'].tolist().index(f'1_3__{g}:TV')
                    err=abs(float(dist[valid].mean())-float(out['scalar'][pos]))
                    assert out['n_valid'][pos]==int(valid.sum())
                    assert err<1e-12
                    pair_checks.append(dict(session_id=sid,object=g,n_valid=int(valid.sum()),max_error=err))
    atomic_json(ROOT/'metadata'/f'{host}_independent_check.json',dict(status='passed',host=host,turns=len(checks),checks=checks,at=now(),
        paired_checks=pair_checks,formula_module_sha256=sha(Path(__file__)),note='Independent scalar, per-position head contribution and real paired formulas; no core metric functions used'))
    print(host,'independent checks passed',len(checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=list(HOSTS),required=True);main(p.parse_args().host)

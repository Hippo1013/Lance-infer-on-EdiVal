#!/usr/bin/env python3
"""Session-equal aggregation and exploration-frozen candidates for EdiVal v2."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import math
import numpy as np
from core import mean_valid
from run import ROOT, sha, atomic_json, save_array, now, TASK


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x


def read_session_arrays(sid):
    for turn in [1,2,3]:
        path=ROOT/'arrays'/sid/f'turn_{turn}.npz'
        with np.load(path,allow_pickle=False) as z:yield turn,{k:z[k] for k in z.files}


class Accumulator:
    def __init__(self):self.stats={}
    def add(self,key,a):
        a=np.asarray(a,float);valid=np.isfinite(a)
        if key not in self.stats:self.stats[key]=[np.zeros(a.shape),np.zeros(a.shape),np.zeros(a.shape,np.int64),np.zeros(a.shape,np.int64)]
        sums,sq,n,below=self.stats[key];sums+=np.where(valid,a,0);sq+=np.where(valid,a*a,0);n+=valid
        # For ratio means, below 50%; for general metrics this is just stored
        # numerically and only used with an explicitly declared reference.
        below+=(valid & (a<.5))
    def export(self,path):
        keys=sorted(self.stats);data={'keys':np.array(keys)}
        for i,k in enumerate(keys):
            sums,sq,n,below=self.stats[k]
            avg=np.divide(sums,n,out=np.full(sums.shape,np.nan),where=n>0)
            var=np.divide(sq,n,out=np.full(sums.shape,np.nan),where=n>0)-avg*avg
            data[f'mean_{i}']=avg.astype(np.float32);data[f'n_{i}']=n
            data[f'std_{i}']=np.sqrt(np.maximum(var,0)).astype(np.float32)
            data[f'below_half_{i}']=np.divide(below,n,out=np.full(sums.shape,np.nan),where=n>0).astype(np.float32)
        save_array(path,data)


def describe(x,clusters,seed=20261010,reps=2000):
    x=np.asarray(x,float);valid=np.isfinite(x);xx=x[valid];cc=np.asarray(clusters)[valid]
    if not len(xx):return dict(n_valid=0,mean=None,ci_low=None,ci_high=None)
    unique=sorted(set(cc));sums=np.array([xx[cc==g].sum() for g in unique]);nn=np.array([(cc==g).sum() for g in unique])
    rng=np.random.default_rng(seed);boot=[]
    for start in range(0,reps,100):
        pick=rng.integers(0,len(unique),size=(min(100,reps-start),len(unique)))
        boot.extend((sums[pick].sum(1)/nn[pick].sum(1)).tolist())
    ci=np.quantile(boot,[.025,.975]);q=np.quantile(xx,[.1,.25,.5,.75,.9])
    k=int(math.floor(len(xx)*.05));trim=np.sort(xx)[k:len(xx)-k] if k else xx
    return dict(n_valid=len(xx),n_clusters=len(unique),mean=float(xx.mean()),median=float(q[2]),p10=float(q[0]),p25=float(q[1]),p75=float(q[3]),p90=float(q[4]),
        ci_low=float(ci[0]),ci_high=float(ci[1]),trimmed_mean=float(trim.mean()),n_total=len(x))


def bootstrap_many(x,clusters):
    """Same cluster draws across metrics; ratio of sampled session sums/counts."""
    x=np.asarray(x,float);unique=sorted(set(clusters));indices={g:np.flatnonzero(np.array(clusters)==g) for g in unique}
    sums=np.stack([np.nansum(x[indices[g]],axis=0) for g in unique]);nn=np.stack([np.isfinite(x[indices[g]]).sum(0) for g in unique])
    rng=np.random.default_rng(20261010);counts=np.zeros((2000,len(unique)))
    for i in range(2000):counts[i]=np.bincount(rng.integers(0,len(unique),size=len(unique)),minlength=len(unique))
    lo=np.full(x.shape[1],np.nan);hi=lo.copy()
    for k in range(0,x.shape[1],40):
        sums_b=counts@sums[:,k:k+40];n_b=counts@nn[:,k:k+40]
        b=np.divide(sums_b,n_b,out=np.full(sums_b.shape,np.nan),where=n_b>0)
        lo[k:k+40],hi[k:k+40]=np.nanquantile(b,[.025,.975],axis=0)
    return lo,hi


def summarize():
    design=json.loads((ROOT/'metadata/design.json').read_text());split=design['split'];sids=sorted(split);clusters=[split[s]['cluster'] for s in sids]
    # Freeze selection using exploration records before any held-out arrays
    # are read by the aggregation process.
    candidates=[]
    targets=[(3,'pair_T1_T2:older_share',.5,'lower'),(3,'pair_I1_combined_I2_combined:older_share',.5,'lower'),
             (3,'I0_vit:left_share',1/64,'higher'),(3,'I0_vae:marker_end_mass',0,'higher')]
    frozen=ROOT/'metadata/frozen_candidates.json'
    if frozen.exists():
        prior=json.loads(frozen.read_text())
        if prior['source_design_sha256']!=sha(ROOT/'metadata/design.json'):raise ValueError('Changed frozen design')
        candidates=prior['candidates']
    else:
        for turn,key,reference,direction in targets:
            selected=[]
            for sid in sids:
                if split[sid]['split']!='exploration':continue
                with np.load(ROOT/'arrays'/sid/f'turn_{turn}.npz',allow_pickle=False) as z:
                    selected.append(z['process'][...,z['names'].tolist().index(key)])
            m=mean_valid(np.stack(selected),0);scored=[]
            for s in range(28):
                for layer in range(33):
                    v=float(mean_valid(m[s:s+3,layer:layer+4]));score=reference-v if direction=='lower' else v-reference
                    scored.append((score,s,layer))
            score,s,layer=max(scored)
            candidates.append(dict(id=f'candidate_{len(candidates)+1}',turn=turn,metric=key,reference=reference,direction=direction,
                step_start=s,step_stop=s+3,layer_start=layer,layer_stop=layer+4,exploration_score=score,
                selection='maximum exploration-only deviation in a fixed contiguous 3x4 window; ties ordered deterministically'))
        atomic_json(frozen,dict(frozen_at=now(),source_design_sha256=sha(ROOT/'metadata/design.json'),candidates=candidates))
    acc=Accumulator();exp=Accumulator();scalar=defaultdict(dict);objects={};nturn=0
    for sid in sids:
        for turn,z in read_session_arrays(sid):
            nturn+=1;names=z['names'].tolist();objects[(sid,turn)]=names
            for mode in ['process','weighted_process','head','step']:
                acc.add(f't{turn}__{mode}',z[mode])
                if split[sid]['split']=='exploration':exp.add(f't{turn}__{mode}',z[mode])
            for i,k in enumerate(names):
                scalar[f't{turn}__{k}'][sid]=float(z['scalar'][i])
                scalar[f't{turn}__weighted__{k}'][sid]=float(z['weighted_scalar'][i])
            for i,k in enumerate(z['profile_names'].tolist()):
                a=z[f'profile_{i}']
                # Variable token supports remain in session-specific arrays.
                if not ':token_' in k:
                    acc.add(f't{turn}__{k}',a)
                    if split[sid]['split']=='exploration':exp.add(f't{turn}__{k}',a)
                    if k.endswith(('head_js','top1_head_share','top4_head_share')):
                        scalar[f't{turn}__{k}'][sid]=float(mean_valid(a))
                if k.endswith(':top_head_identity'):
                    for layer in range(36):
                        for head in range(16):scalar[f't{turn}__{k}__l{layer}__h{head}'][sid]=float((a[:,layer]==head).mean())
        for p in sorted((ROOT/'paired'/sid).glob('*.npz')):
            with np.load(p,allow_pickle=False) as z:
                for mode in ['process','head','step']:acc.add('paired__'+p.stem+'__'+mode,z[mode])
                for i,k in enumerate(z['names'].tolist()):
                    scalar['paired__'+k][sid]=float(z['scalar'][i]);scalar['paired__weighted__'+k][sid]=float(z['weighted_scalar'][i])
    if nturn!=1716:raise ValueError('Incomplete coverage')
    for turn in [1,2,3]:
        a=objects[sids[0],turn]
        if any(objects[sid,turn]!=a for sid in sids):raise ValueError('Metric columns differ')
    atomic_json(ROOT/'metadata/metric_names.json',{str(t):objects[sids[0],t] for t in [1,2,3]})
    acc.export(ROOT/'summaries/full_atlas.npz');exp.export(ROOT/'summaries/exploration_atlas.npz')
    keys=sorted(scalar)
    # Head identity frequencies are written separately, avoiding thousands of
    # redundant bootstrap intervals for a descriptive support chart.
    identity_keys=[k for k in keys if '__l' in k and ':top_head_identity' in k]
    keys=[k for k in keys if k not in identity_keys]
    x=np.array([[scalar[k].get(sid,np.nan) for k in keys] for sid in sids])
    lo,hi=bootstrap_many(x,clusters)
    summaries=[]
    for i,k in enumerate(keys):
        v=x[:,i];valid=v[np.isfinite(v)];q=np.quantile(valid,[.1,.25,.5,.75,.9]) if len(valid) else [np.nan]*5
        trim=int(len(valid)*.05);vv=np.sort(valid)[trim:len(valid)-trim] if trim else valid
        summaries.append(clean(dict(metric=k,mean=mean_valid(v),median=q[2],p10=q[0],p25=q[1],p75=q[3],p90=q[4],ci_low=lo[i],ci_high=hi[i],
            n_valid=len(valid),n_total=572,n_clusters=len(set(np.array(clusters)[np.isfinite(v)])),trimmed_mean=mean_valid(vv),
            below_half=float((valid<.5).mean()) if len(valid) else None,bootstrap_reps=2000)))
    atomic_json(ROOT/'summaries/scalar_summary.json',summaries)
    save_array(ROOT/'summaries/session_scalars.npz',{'session_ids':np.array(sids),'metric_names':np.array(keys),'values':x,'clusters':np.array(clusters),'splits':np.array([split[s]['split'] for s in sids])})
    with gzip.open(ROOT/'tables/session_scalars.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(['session_id','original_image_cluster','split','metric','value'])
        for sid in sids:
            for k in keys:w.writerow([sid,split[sid]['cluster'],split[sid]['split'],k,'' if not np.isfinite(scalar[k].get(sid,np.nan)) else scalar[k][sid]])
    identity={k:float(np.mean(list(scalar[k].values()))) for k in identity_keys}
    atomic_json(ROOT/'summaries/fixed_head_win_rates.json',identity)
    # The selected windows are immutable; the following reads evaluate them.
    candidate_rows=[]
    for c in candidates:
        values=[];weighted=[];sens=[]
        key=c['metric'];turn=c['turn'];idx=objects[sids[0],turn].index(key)
        for sid in sids:
            with np.load(ROOT/'arrays'/sid/f'turn_{turn}.npz',allow_pickle=False) as z:
                region=(slice(c['step_start'],c['step_stop']),slice(c['layer_start'],c['layer_stop']))
                # Main candidate: equal heads per process position and equal
                # positions in the fixed window. Weighting is a supplementary
                # within-position check with explicitly different semantics.
                values.append(float(mean_valid(z['process'][region+(idx,)])))
                weighted.append(float(mean_valid(z['weighted_process'][region+(idx,)])))
                skey=key+'_sensitivity'
                sens.append(float(mean_valid(z['process'][region+(objects[sid,turn].index(skey),)])) if skey in objects[sid,turn] else np.nan)
        for label in ['exploration','confirmation','all']:
            mask=np.array([label=='all' or split[s]['split']==label for s in sids]);v=np.array(values)[mask];cc=np.array(clusters)[mask]
            ref=c['reference'];threshold=.01 if key.endswith('share') else (1e-4 if key.endswith('mass') else .01)
            signed=v-ref
            counts={'lower':float((signed < -threshold).mean()),'near':float((np.abs(signed)<=threshold).mean()),'higher':float((signed>threshold).mean())}
            row=dict(c,split=label,summary=describe(v,cc),weighted_within_position_mean=float(mean_valid(np.array(weighted)[mask])),sensitivity_mean=float(mean_valid(np.array(sens)[mask])),
                coverage_threshold=threshold,coverage=counts)
            candidate_rows.append(clean(row))
        v=np.array(values);typical=int(np.nanargmin(abs(v-np.nanmedian(v))));strong=int(np.nanargmin(v) if c['direction']=='lower' else np.nanargmax(v))
        reverse=np.flatnonzero((v>=c['reference']) if c['direction']=='lower' else (v<=c['reference']))
        opposite=int(reverse[np.argmax(v[reverse]) if c['direction']=='lower' else np.argmin(v[reverse])]) if len(reverse) else None
        c['cases']={'typical':sids[typical],'strong':sids[strong],'opposite':sids[opposite] if opposite is not None else None}
        # Case identities attach to a separate file, preserving frozen definitions.
    atomic_json(ROOT/'summaries/candidate_review.json',dict(frozen_candidates_sha256=sha(frozen),rows=candidate_rows,cases=candidates))
    atomic_json(ROOT/'aggregate_status.json',dict(task_id=TASK,status='summarized',sessions=572,turns=1716,metrics=len(keys),at=now()))


if __name__=='__main__':
    (ROOT/'tables').mkdir(parents=True,exist_ok=True)
    summarize()

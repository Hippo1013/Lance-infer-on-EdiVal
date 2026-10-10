"""Separate direct recomputation. Does not import production metrics/aggregation."""
import csv, gzip, hashlib, json, math, pathlib
from collections import defaultdict
import numpy as np
from common import *

NAMES=['support_mass','entropy_nats','normalized_entropy','effective_count','effective_fraction','top1_share','top3_share','top5_share','top10pct_share','n90','n90_fraction']

def read_table(name):
    with gzip.open(ROOT/'tables'/f'{name}.csv.gz','rt',encoding='utf8',newline='') as f:yield from csv.DictReader(f)

def number(x):return None if x=='' else float(x)

def reference(x):
    x=np.asarray(x,dtype=np.float64);N=x.shape[-1];M=np.add.reduce(x,axis=-1)
    pos=M>0;p=np.zeros_like(x);p[pos]=x[pos]/M[pos,None]
    # Alternate entropy formula: log(M) - sum(x*log(x))/M.
    xl=np.zeros_like(x);sel=x>0;xl[sel]=x[sel]*np.log(x[sel]);H=np.zeros(len(x));H[pos]=np.log(M[pos])-np.add.reduce(xl,axis=-1)[pos]/M[pos]
    H[np.abs(H)<1e-14]=0
    ordered=-np.sort(-p,axis=-1);c=np.cumsum(ordered,axis=-1)
    n90=1+np.sum(c < .9-1e-12,axis=-1);n90=np.minimum(n90,N)
    top=[]
    for k in [1,min(3,N),min(5,N),math.ceil(.1*N)]:top.append(np.add.reduce(ordered[:,:k],axis=-1))
    values=np.stack([M,H,H/math.log(N) if N>1 else np.zeros(len(M)),np.exp(H),np.exp(H)/N,*top,n90,n90/N],axis=-1)
    valid=np.tile(pos[:,None],(1,len(NAMES)));valid[:,0]=True
    if N==1:valid[:,2]=False
    values[~valid]=0
    return values,valid,p,pos

def run(temp):
    selection=json.loads((ROOT/'metadata/sample_selection.json').read_text());samples=set(selection['sessions'])
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:inputs=[json.loads(line) for line in f]
    groups={(r['source_id'],r['group_name']):r for r in read_table('group_turn') if r['session_id'] in samples}
    supports={(r['source_id'],r['support_id']):r for r in read_table('concentration_turn') if r['session_id'] in samples}
    profiles={}
    for name in ['text_token_profiles','image_region_profiles']:
        for r in read_table(name):
            if r['session_id'] not in samples:continue
            cell=int(r['token_ordinal']) if name.startswith('text') else int(r['row'])*8+int(r['col'])
            profiles[(r['source_id'],r['support_id'],cell)]=r
    trajectory_refs={};errors=defaultdict(float);checked=[];positions=0
    def close(field,actual,expected,tol=2e-10):
        if expected is None:
            assert actual is None,(field,actual,expected);return
        assert actual is not None
        delta=abs(actual-expected);errors[field]=max(errors[field],delta)
        assert delta<=tol,(field,actual,expected,delta,tol)
    for item in inputs:
        if item['session_id'] not in samples:continue
        sid=item['session_id'];t=item['generation_turn'];source=item['source_id'];path=pathlib.Path(item['original_path']);remote=item['host']=='a800_1'
        if remote:
            path=pathlib.Path(temp)/('audit_'+sid.replace('/','__')+f'_{t}.npz')
            download(f'/attention/{sid}/turn_{t}.attention.npz',path,item['bytes'],item['sha256'])
        try:
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
            with np.load(path,allow_pickle=False) as z, np.load(ROOT/'arrays'/sid/f'turn_{t}.npz',allow_pickle=False) as derived:
                names=z['group_names'].tolist();means=z['group_stats'][...,z['stat_names'].tolist().index('mean')].astype(np.float64).reshape(17280,-1).mean(0)
                counts=z['group_token_counts'];N=int(sum(counts))
                for g,v,n in zip(names,means,counts):
                    row=groups[(source,g)];close('group_mass',number(row['mass']),float(v));close('per_token_mass',number(row['per_token_mass']),float(v/n));close('enrichment',number(row['enrichment']),float(v*N/n))
                    trajectory_refs[(sid,t,g)]=(v,v/n,v*N/n)
                for i in range(t):
                    a,b=[names.index(f'I{i}_{m}') for m in ['vit','vae']];v=means[a]+means[b];n=counts[a]+counts[b]
                    trajectory_refs[(sid,t,f'I{i}_combined')]=(v,v/n,v*N/n)
                ids=derived['support_ids'].tolist();dmetrics=derived['metric_names'].tolist();assert dmetrics==NAMES
                for si,support in enumerate(ids):
                    if support.startswith('T'):
                        select=z['text_token_turns']==int(support[1:]);x=z['text_mean'][...,select].astype(np.float64).reshape(17280,-1);cells=np.arange(x.shape[1])
                    else:
                        im=int(support[1:].split('_')[0]);mi=int(support.endswith('vae'));rc=z['region_token_counts'][im,mi].ravel();cells=np.flatnonzero(rc>0)
                        x=z['region_mean'][...,im,mi,:,:].astype(np.float64).reshape(17280,64)[:,cells]
                    v,valid,p,pos=reference(x);positions+=len(x);row=supports[(source,support)]
                    assert np.array_equal(derived['metric_valid'][...,si,:].reshape(17280,-1),valid)
                    assert np.array_equal(derived['positive_mass'][...,si].ravel(),pos)
                    assert np.array_equal(derived['low_mass'][...,si].ravel(),pos & (v[:,0]<=1e-6))
                    dv=derived['metric_values'][...,si,:].reshape(17280,-1).astype(np.float64)
                    assert np.all(np.abs(dv-v)<=np.maximum(1e-7,np.abs(v)*6.1e-8)), ('FP32 arrays',sid,t,support,float(np.abs(dv-v).max()))
                    errors['fp32_array']=max(errors['fp32_array'],float(np.abs(dv-v).max()))
                    for k,metric in enumerate(NAMES):
                        close(metric,number(row[metric]),float(v[valid[:,k],k].mean()) if valid[:,k].any() else None)
                        assert int(row[metric+'_n_valid'])==int(valid[:,k].sum())
                        sens=valid[:,k] & pos & (v[:,0]>1e-6)
                        close(metric+'_sensitivity',number(row[metric+'_sensitivity']),float(v[sens,k].mean()) if sens.any() else None)
                        assert int(row[metric+'_sensitivity_n_valid'])==int(sens.sum())
                    close('zero_mass_fraction',number(row['zero_mass_fraction']),float((~pos).mean()))
                    close('low_mass_fraction',number(row['low_mass_fraction']),float((pos & (v[:,0]<=1e-6)).mean()))
                    for k,cell in enumerate(cells):
                        pr=profiles[(source,support,int(cell))];raw=float(x[:,k].mean());conditional=float(p[pos,k].mean()) if pos.any() else None;pooled=raw/float(v[:,0].mean()) if pos.any() else None
                        close('mean_raw_mass',number(pr['mean_raw_mass']),raw);close('mean_conditional_share',number(pr['mean_conditional_share']),conditional);close('pooled_conditional_share',number(pr['pooled_conditional_share']),pooled)
                        if 'region_token_count' in pr:close('region_density',number(pr['per_token_raw_mass']),raw/int(pr['region_token_count']))
            checked.append(dict(source_id=source,host=item['host'],sha256=item['sha256'],positions=17280))
        finally:
            if remote:path.unlink(missing_ok=True)
    pair_count=0
    for r in read_table('paired_changes'):
        if r['session_id'] not in samples:continue
        idx=int(r['object_index']);group=f'T{idx}' if r['object_type']=='text' else f'I{idx}_{r["modality"]}'
        a=trajectory_refs[(r['session_id'],int(r['turn_from']),group)];b=trajectory_refs[(r['session_id'],int(r['turn_to']),group)]
        for k,name in enumerate(['delta_mass','delta_per_token_mass','delta_enrichment']):close('paired_'+name,number(r[name]),float(b[k]-a[k]))
        pair_count+=1
    assert len(checked)==18 and len(samples)==6 and pair_count==96
    # Independent whole-table session weighting and selected bootstrap recomputation.
    cluster_map={}
    with gzip.open(ROOT/'metadata/source_image_clusters.csv.gz','rt') as f:
        for r in csv.DictReader(f):cluster_map[r['session_id']]=r['original_image_hash']
    tables={name:list(read_table(name)) for name in ['group_turn','image_combined_turn','category_turn','category_modality_turn','concentration_turn','object_trajectories','paired_changes']}
    summary_checks=0;bootstrap_checks=[]
    def point_check(row,rs,field,role=False):
        by=defaultdict(lambda:defaultdict(list))
        for r in rs:by[r['session_id']][r.get('generation_turn','paired')].append(number(r[field]))
        values={}
        for s,turns in by.items():
            tv=[]
            for vv in turns.values():
                valid=[v for v in vv if v is not None]
                if valid:tv.append(math.fsum(valid)/len(valid))
            values[s]=math.fsum(tv)/len(tv) if tv else None
        vx=[v for v in values.values() if v is not None]
        close('summary_point',number(row['mean']),math.fsum(vx)/len(vx) if vx else None)
        assert int(row['n_valid'])==len(vx) and int(row['n_sessions'])==len(values)
        assert int(row['n_clusters'])==len({cluster_map[s] for s,v in values.items() if v is not None})
        close('summary_std',number(row['std']),float(np.std(vx,ddof=1)) if len(vx)>1 else None)
        return values
    for name in ['composition_summary','trajectory_summary','paired_summary','concentration_summary']:
        for r in read_table(name):
            if name=='composition_summary':
                rs=tables[r['table']];rs=[x for x in rs if all(not r.get(k) or x.get(k)==r[k] for k in ['group_name','category','modality'])];field=r['metric']
            elif name=='trajectory_summary':
                rs=[x for x in tables['object_trajectories'] if all(x[k]==r[k] for k in ['object_type','object_index','modality'])];field=r['metric']
            elif name=='paired_summary':
                rs=[x for x in tables['paired_changes'] if all(x[k]==r[k] for k in ['object_type','object_index','modality','turn_from','turn_to'])];field=r['metric']
            else:
                group=r['grouping'];rs=[x for x in tables['concentration_turn'] if x['support_type']==r['support_type'] and x['modality']==r['modality'] and x[group]==r[group]]
                field=r['metric']+('_sensitivity' if r['variant']=='mass_gt_1e-6' else '')
            if r.get('generation_turn') and r['generation_turn']!='overall':rs=[x for x in rs if x['generation_turn']==r['generation_turn']]
            values=point_check(r,rs,field);summary_checks+=1
            # Three representative intervals, same declared random stream, alternate weighted-count calculation.
            target=(name=='composition_summary' and r.get('category')=='history_instructions' and r['generation_turn']=='overall' and r['metric']=='per_token_mass') or (name=='paired_summary' and r['object_type']=='image' and r['object_index']=='0' and r['modality']=='combined' and r['turn_from']=='1' and r['turn_to']=='3' and r['metric']=='delta_mass') or (name=='concentration_summary' and r['grouping']=='object_role' and r['object_role']=='history_instruction' and r['generation_turn']=='overall' and r['metric']=='entropy_nats' and r['variant']=='main')
            if target:
                if name=='composition_summary':key=(r['table'],(r['category'],),r['metric'],'overall')
                elif name=='paired_summary':key=('paired',(r['object_type'],int(r['object_index']),r['modality'],int(r['turn_from']),int(r['turn_to'])),r['metric'])
                else:key=('concentration',r['grouping'],(r['support_type'],r['modality'],r['object_role']),r['metric'],r['variant'],'overall')
                gs=defaultdict(list)
                for s,v in sorted(values.items()):
                    if v is not None:gs[cluster_map[s]].append(v)
                ordered=sorted(gs);sums=np.array([math.fsum(gs[g]) for g in ordered]);counts=np.array([len(gs[g]) for g in ordered])
                salt=int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8],'little');rng=np.random.default_rng(np.random.SeedSequence([20261008,salt]));boot=[]
                for _ in range(2000):
                    freq=np.bincount(rng.integers(0,len(ordered),len(ordered)),minlength=len(ordered));boot.append(float(np.dot(freq,sums)/np.dot(freq,counts)))
                lo,hi=np.percentile(boot,[2.5,97.5]);close('bootstrap_ci',number(r['ci_low']),float(lo));close('bootstrap_ci',number(r['ci_high']),float(hi));bootstrap_checks.append(dict(table=name,key=str(key)))
    assert len(bootstrap_checks)==3
    atomic_json(ROOT/'validation/independent_samples.json',dict(task_id=TASK,status='passed',at=now(),implementation='independent.py; separate formulas and reductions, no production metric/aggregation imports',
        samples=checked,input_npz_count=18,support_positions=positions,paired_rows_checked=pair_count,summary_rows_checked=summary_checks,bootstrap_checks=bootstrap_checks,max_absolute_errors=dict(errors)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--temp',required=True);a=p.parse_args();run(a.temp);print('Independent direct recomputation passed')

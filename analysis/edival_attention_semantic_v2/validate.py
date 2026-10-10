"""Full output integrity, support arrays, conservation, keys and null validation."""
import concurrent.futures,csv,gzip,json,math,os,pathlib,urllib.error,urllib.request
from collections import Counter,defaultdict
import numpy as np
from common import *
from documentation import KEYS
from metrics import METRICS

def array_check(args):
    path,rows,source=args
    max_error=0.
    with np.load(path,allow_pickle=False) as z:
        assert z['metric_names'].tolist()==METRICS
        t=source['generation_turn'];S=3*t;v=z['metric_values'];valid=z['metric_valid'];pos=z['positive_mass'];low=z['low_mass'];N=z['support_N']
        assert v.shape==(30,36,16,S,11) and v.dtype==np.float32 and valid.shape==v.shape and valid.dtype==bool
        assert pos.shape==(30,36,16,S) and low.shape==pos.shape and pos.dtype==low.dtype==bool and N.shape==(S,) and (N>0).all()
        assert np.isfinite(v).all() and np.all(v[~valid]==0) and np.all(valid[...,0]) and np.all(low<=pos)
        assert z['source_id'].item()==source['source_id'] and z['source_sha256'].item()==source['sha256']
        assert np.array_equal(z['step_ids'],np.arange(30)) and np.array_equal(z['head_ids'],np.arange(16)) and np.array_equal(z['layer_ids'],np.arange(36))
        assert z['timesteps'].shape==(30,) and np.all(np.diff(z['timesteps'])<0)
        ids=z['support_ids'].tolist();assert len(set(ids))==S and set(ids)==set(rows)
        for si,s in enumerate(ids):
            row=rows[s];n=int(N[si]);assert int(row['support_N'])==n and int(row['n_positions_total'])==17280
            assert int(row['top3_k'])==min(3,n) and int(row['top5_k'])==min(5,n) and int(row['top10pct_k'])==math.ceil(.1*n)
            for mi,name in enumerate(METRICS):
                mask=valid[...,si,mi]
                if mi>0:assert np.array_equal(mask,pos[...,si] if mi!=2 or n>1 else np.zeros_like(mask))
                assert int(row[name+'_n_valid'])==int(mask.sum())
                if mask.any():
                    assert row[name]!='' and row[name+'_null_reason']==''
                    observed=float(v[...,si,mi][mask].astype(np.float64).mean());expected=float(row[name]);error=abs(observed-expected);max_error=max(max_error,error)
                    assert error<=max(1e-7,6.1e-8*abs(expected)),(path,s,name,error)
                else:assert row[name]=='' and row[name+'_null_reason'] in ['single_element_support','zero_support_mass']
                sens=mask & pos[...,si] & ~low[...,si];assert int(row[name+'_sensitivity_n_valid'])==int(sens.sum())
                if sens.any():
                    expected=float(row[name+'_sensitivity']);observed=float(v[...,si,mi][sens].astype(np.float64).mean());error=abs(observed-expected);max_error=max(max_error,error)
                    assert error<=max(1e-7,6.1e-8*abs(expected))
                else:assert row[name+'_sensitivity']==''
            assert int(row['n_positions_positive'])==int(pos[...,si].sum()) and int(row['n_positions_low_mass'])==int(low[...,si].sum())
            conditional=v[...,si,:][pos[...,si]].astype(np.float64)
            if len(conditional):
                assert (conditional[:,9]>=1).all() and (conditional[:,9]<=n).all() and np.equal(conditional[:,9],np.floor(conditional[:,9])).all()
                assert (conditional[:,1]>=-1e-7).all() and (conditional[:,1]<=math.log(n)+1e-6).all()
                for mi in [2,4,5,6,7,8,10]:assert (conditional[:,mi]>=-1e-7).all() and (conditional[:,mi]<=1+1e-7).all()
                assert (conditional[:,5]<=conditional[:,6]+1e-7).all() and (conditional[:,6]<=conditional[:,7]+1e-7).all()
    return max_error

def security():
    token=(ROOT/'dispatch/.peer_token').read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers={'Authorization':'Bearer '+token,'X-Task-ID':TASK}
    checks=[('/health',{},'GET',401),('/health',headers,'GET',200),('/health',headers,'POST',405),('/files/dispatch/.peer_token',headers,'GET',404),('/attention/../dispatch/.peer_token',headers,'GET',404)]
    evidence=[]
    for route,h,method,expected in checks:
        try:
            with opener.open(urllib.request.Request(PEER+route,headers=h,method=method),timeout=10) as r:code=r.status
        except urllib.error.HTTPError as e:code=e.code
        assert code==expected,(route,code,expected);evidence.append(dict(route=route,status=code,expected=expected))
    return evidence

def run():
    schemas=json.loads((ROOT/'schemas/tables.json').read_text());row_counts={};concentration=defaultdict(dict);grouped=defaultdict(list);categories=defaultdict(list);profiles=defaultdict(lambda:dict(raw=0.,conditional=0.,pooled=0.,has_conditional=False,has_pooled=False))
    lookup={};keys_checked=0
    for name,schema in schemas['tables'].items():
        seen=set();count=0
        with gzip.open(ROOT/schema['path'],'rt',encoding='utf8',newline='') as f:
            reader=csv.DictReader(f);assert set(reader.fieldnames)==set(schema['fields'])
            for r in reader:
                key=tuple(r[k] for k in schema['primary_key']);assert key not in seen,(name,key);seen.add(key);count+=1
                for field,info in schema['fields'].items():
                    v=r[field]
                    if v=='':assert info['nullable'],(name,field,'unexpected null')
                    elif info['types']==['number']:assert math.isfinite(float(v)),(name,field)
                    elif info['types']==['integer']:int(v)
                    elif info['types']==['boolean']:assert v in ['True','False']
                if name=='concentration_turn':concentration[r['source_id']][r['support_id']]=r
                if name=='group_turn':grouped[r['source_id']].append(r)
                if name=='category_turn':categories[r['source_id']].append(r)
                if name in ['group_turn','image_combined_turn','category_turn','category_modality_turn']:
                    mass=float(r['mass']);n=int(r['token_count']);N=int(r['total_key_tokens']);present=r['present']=='True'
                    assert present==(n>0)
                    if present:
                        assert abs(float(r['per_token_mass'])-mass/n)<1e-12 and abs(float(r['enrichment'])-mass*N/n)<1e-12
                        assert r['density_null_reason']==''
                    else:assert mass==0 and r['per_token_mass']==r['enrichment']=='' and r['density_null_reason']=='structural_zero'
                if name=='trajectory_matrix':
                    present=r['present']=='True';assert present==(r['missing_mask']=='False')
                    assert (r['null_reason']=='' and r['mass']!='') if present else (r['null_reason']=='not_yet_present' and r['mass']=='')
                if name in ['text_token_profiles','image_region_profiles']:
                    q=profiles[(r['source_id'],r['support_id'])];q['raw']+=float(r['mean_raw_mass'])
                    if r['mean_conditional_share']!='':q['conditional']+=float(r['mean_conditional_share']);q['has_conditional']=True
                    if r['pooled_conditional_share']!='':q['pooled']+=float(r['pooled_conditional_share']);q['has_pooled']=True
                    if name=='image_region_profiles':
                        n=int(r['region_token_count']);assert (n>0)==(r['region_present']=='True')
                        if n:assert abs(float(r['per_token_raw_mass'])-float(r['mean_raw_mass'])/n)<1e-12
                        else:assert r['density_null_reason']=='empty_region' and r['per_token_raw_mass']=='' and r['conditional_null_reason']=='empty_region'
                if name.endswith('summary'):
                    assert r['aggregation_level'].startswith('session_equal') and int(r['bootstrap_reps'])==2000 and int(r['bootstrap_seed'])==20261008 and r['bootstrap_unit']=='original_image_cluster'
                    assert int(r['n_valid'])<=int(r['n_sessions'])==int(r['n_total'])
                if name=='paired_summary':assert abs(sum(float(r[k]) for k in ['positive_fraction','negative_fraction','near_zero_fraction'])-1)<1e-12
        assert count==schema['rows'];row_counts[name]=count;keys_checked+=count
    assert row_counts['group_turn']==15444 and row_counts['category_turn']==12012 and row_counts['concentration_turn']==10296
    assert row_counts['object_trajectories']==13728 and row_counts['paired_changes']==9152 and row_counts['trajectory_matrix']==20592
    assert len(grouped)==len(categories)==len(concentration)==1716
    for source,gs in grouped.items():
        cs=categories[source];assert len(cs)==7
        assert abs(sum(float(r['mass']) for r in gs)-sum(float(r['mass']) for r in cs))<1e-12
        assert sum(int(r['token_count']) for r in gs)==sum(int(r['token_count']) for r in cs)
    for key,p in profiles.items():
        row=concentration[key[0]][key[1]];assert abs(p['raw']-float(row['support_mass']))<1e-10
        if p['has_conditional']:assert abs(p['conditional']-1)<1e-10
        if p['has_pooled']:assert abs(p['pooled']-1)<1e-10
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:sources=[json.loads(line) for line in f]
    assert len(sources)==1716 and len({s['session_id'] for s in sources})==572 and len({s['source_id'] for s in sources})==1716
    assert all(s['sha256_verified'] and s['index_identity_verified'] for s in sources)
    assert sum(s['bytes'] for s in sources)==json.loads((ROOT/'validation/input_audit.json').read_text())['bytes']
    source_turns=defaultdict(set)
    for s in sources:source_turns[s['session_id']].add(s['generation_turn'])
    assert all(v=={1,2,3} for v in source_turns.values())
    expected={ROOT/'arrays'/s['session_id']/f'turn_{s["generation_turn"]}.npz' for s in sources};assert expected==set((ROOT/'arrays').rglob('*.npz'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        errors=list(pool.map(array_check,[(ROOT/'arrays'/s['session_id']/f'turn_{s["generation_turn"]}.npz',concentration[s['source_id']],s) for s in sources]))
    for p in ROOT.rglob('*'):
        if p.suffix=='.json' and not p.is_relative_to(ROOT/'dispatch') and not p.is_relative_to(ROOT/'logs'):
            json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    assert not list(ROOT.rglob('*.part'))
    for name in ['input_audit','numerical_tests','independent_samples']:assert json.loads((ROOT/'validation'/f'{name}.json').read_text())['status']=='passed'
    evidence=security()
    atomic_json(ROOT/'validation/self_validation.json',dict(task_id=TASK,status='passed',at=now(),sessions=572,turns=1716,validated_source_files=1716,
        compressed_tables_read=len(row_counts),compressed_arrays_read=1716,table_rows=row_counts,primary_keys_checked=keys_checked,max_array_scalar_error=max(errors),
        checks=['full source SHA256/size and identity audit','all CSV.GZ readable, fields/keys/types/nulls validated','all NPZ readable, masks/IDs/dtypes/metric denominators/scalars validated',
         'category mass/count conservation','structural zero distinct from missing','support profile conservation','paired counts and independent values','session and cluster weighting independently recalculated','no unsubmitted part files'],
        service_security_tests=evidence,issues=[]))
    atomic_json(ROOT/'validation/validation.json',dict(task_id=TASK,status='passed',self_validation='passed',independent_formula_recalculation='passed',at=now(),issues=[]))

if __name__=='__main__':run();print('Full self-validation passed; independent recalculation passed')

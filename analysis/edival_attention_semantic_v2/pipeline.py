"""Full receipt-bound EdiVal offline analysis. Never imports inference code."""
import argparse, concurrent.futures, csv, gzip, json, os, pathlib, platform, sys, time, traceback
from collections import Counter, defaultdict
import numpy as np
from common import *
from metrics import METRICS, calculate, scalar_rows
from aggregation import describe, equal_average, SEED, REPS

TABLES=['group_turn','image_combined_turn','category_turn','category_modality_turn','object_trajectories','concentration_turn','text_token_profiles','image_region_profiles','within_type_pairs','image_marker_turn']
CATEGORIES=['current_instruction','history_instructions','original_image','history_images','target_image','context_other','generation_markers']

def write_gz_jsonl(path,rows):
    path=pathlib.Path(path);part=path.with_name(path.name+'.part')
    with gzip.open(part,'wt',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(part,path)

def write_csv(name,rows):
    path=ROOT/'tables'/ (name+'.csv.gz');part=path.with_name(path.name+'.part')
    if not rows:raise ValueError('Empty table '+name)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(part,'wt',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for v in r.values():
                if isinstance(v,float) and not np.isfinite(v):raise ValueError('nonfinite CSV')
            w.writerow(r)
    os.replace(part,path)

def prepare():
    assert platform.node()==HOSTS['a800_0']
    old=PROJECT/'outputs/attention_analysis/edival_descriptive_20261008_v1'
    collect=PROJECT/'outputs/attention_recollection'/TASK
    with gzip.open(old/'metadata/source_manifest.jsonl.gz','rt') as f:prior=[json.loads(line) for line in f]
    assert len(prior)==1716
    peer=peer_json('/export'); assert peer['sessions']==293 and peer['turns']==879 and peer['status']=='passed'
    peer_by={x['source_id']:x for x in peer['items']}
    local={}
    for stage in ['full_gpu0','full_gpu1']:
        c=json.loads((collect/stage/'completion.json').read_text());assert c['status']=='passed' and c['all_outputs_exact']
        for rec in c['records']:
            sid=rec['session_id'];t=rec['turn'];p=collect/stage/'run'/sid.split('/')[-1]/'observed'/f'turn_{t}.json'
            local[f'{sid}/turn_{t}']={'json':json.loads(p.read_text()),'json_path':str(p),'attention_path':str(p.with_name(f'turn_{t}.attention.npz'))}
    assert len(local)==837
    items=[];clusters={};sessions=set()
    for olditem in prior:
        x=(local if olditem['host']=='a800_0' else peer_by)[olditem['source_id']]
        j=x['json'];b=j['backend'];a=b['attention'];t=j['turn'];sid=j['session_id']
        assert j['original_turn_json_sha256']==olditem['json_sha256'] and j['original_attention_sha256']==olditem['sha256']
        assert j['input_hashes']==b['image_hashes']==olditem['input_hashes'] and j['output_hash']==olditem['output_hash']
        assert j['session_fingerprint']==olditem['session_fingerprint'] and b['protocol']==olditem['input_protocol']
        assert b['cache_identity_audit']['status']=='passed' and a['version']=='target-token-region-stats-v2' and a['branch']=='positive'
        if olditem['host']=='a800_1':
            jp=ROOT/'metadata/peer_turns'/sid/f'turn_{t}.json';jp.parent.mkdir(parents=True,exist_ok=True);atomic_json(jp,j)
            n=x['bytes']
        else:jp=pathlib.Path(x['json_path']);n=pathlib.Path(x['attention_path']).stat().st_size
        item=dict(olditem,original_path=x['attention_path'],resolved_path=x['attention_path'],bytes=n,sha256=a['sha256'],json_path=str(jp),json_sha256=sha(jp),
            attention_version=a['version'],source_run=TASK,source_original_attention_sha256=olditem['sha256'],source_original_turn_json_sha256=olditem['json_sha256'],sha256_verified=False)
        items.append(item);clusters[sid]=j['input_hashes'][0];sessions.add(sid)
    assert len(sessions)==572 and len(clusters)==572 and len(set(clusters.values()))==570
    samples=[]
    for host in HOSTS:
        sids=sorted({x['session_id'] for x in items if x['host']==host});samples.extend([sids[0],sids[len(sids)//2],sids[-1]])
    shutil_path=old/'metadata/source_image_clusters.csv.gz'
    import shutil
    shutil.copy2(shutil_path,ROOT/'metadata/source_image_clusters.csv.gz')
    write_gz_jsonl(ROOT/'metadata/source_manifest.jsonl.gz',items)
    atomic_json(ROOT/'metadata/sample_selection.json',dict(task_id=TASK,sessions=samples,turns=18,method='fixed first/middle/last session per source host'))
    atomic_json(ROOT/'metadata/run.json',dict(task_id=TASK,input_sessions=572,input_turns=1716,attention_version='target-token-region-stats-v2',bootstrap_seed=SEED,bootstrap_reps=REPS,
        bootstrap_unit='original_image_cluster',n_original_image_clusters=570,calculation_dtype='float64',array_storage_dtype='float32',all_original_outputs_exact=True,
        point_weighting='equal sessions; equal available objects within role; equal present turns within session',prior_manifest_sha256=sha(old/'metadata/source_manifest.jsonl.gz')))
    return items,clusters,samples

def layout_audit(z,j):
    b=j['backend'];meta=b['attention'];trace=b['positive_trace'];t=j['turn']
    names=['context_other'];N=sum(meta['group_token_counts']);labels=np.zeros(N,dtype=np.int64);fine=np.full(N,-1,dtype=np.int64)
    context=b['positive_kv_tokens']
    marker_end=context+meta['group_token_counts'][-2]
    assert N==marker_end+meta['target_query_count']
    text_keys=[];text_ids=[];text_turns=[];offsets=[];geometry=[];occupied=np.zeros(context,bool)
    for x in trace:
        lo,hi=x['tokens'];assert 0<=lo<=hi<=context
        assert not occupied[lo:hi].any();occupied[lo:hi]=True
        if x['kind'] in ['vit','vae']:
            name=f'I{x["image_index"]}_{x["kind"]}';names.append(name);labels[lo:hi]=len(names)-1
            a,c=x['spatial_tokens'];h,w=x['grid_hw'];expected=(lo+1,hi-1) if x['kind']=='vit' else (lo+2,hi)
            assert (a,c)==expected and c-a==h*w and x['cache_identity_verified']
            markers=[lo,hi-1] if x['kind']=='vit' else [lo,lo+1]
            assert x['marker_key_positions']==markers
            labels[markers]=0
            yy,xx=np.indices((h,w));cell=((2*yy+1)*8//(2*h))*8+((2*xx+1)*8//(2*w))
            base=(2*x['image_index']+(x['kind']=='vae'))*64;fine[a:c]=base+cell.ravel();geometry.append(x)
        elif x.get('role') in ['current','history']:
            name=f'T{x["turn"]}';names.append(name);labels[lo:hi]=len(names)-1
            assert hi-lo==len(x['token_ids'])==len(x['offsets'])>0
            for off in x['offsets']:assert 0<=off[0]<=off[1]<=len(x['text'])
            text_keys.extend(range(lo,hi));text_ids.extend(x['token_ids']);text_turns.extend([x['turn']]*(hi-lo));offsets.extend(x['offsets'])
    names+=['generation_markers','target_image'];labels[context:marker_end]=len(names)-2;labels[marker_end:]=len(names)-1
    assert len(set(names))==len(names)==3*t+3
    assert names==z['group_names'].tolist()==meta['groups']
    fine[text_keys]=t*128+np.arange(len(text_keys))
    marker_positions=[p for x in geometry for p in x['marker_key_positions']]
    assert np.array_equal(z['image_marker_key_positions'],marker_positions)
    assert z['image_marker_mean'].shape==(30,36,16,t,2,2)
    assert np.isfinite(z['image_marker_mean']).all() and (z['image_marker_mean']>=0).all()
    expected={'group_token_counts':np.bincount(labels,minlength=len(names)),'token_group_ids':labels,'fine_key_channel':fine,
        'text_key_positions':text_keys,'text_token_ids':text_ids,'text_token_turns':text_turns,'text_token_offsets':offsets,
        'region_token_counts':np.bincount(fine[fine>=0],minlength=t*128+len(text_keys))[:t*128].reshape(t,2,8,8)}
    for k,val in expected.items():assert z[k].dtype==np.int64 and np.array_equal(z[k],val),(k,j['backend']['session_id'],t)
    assert json.loads(str(z['image_geometry_json']))==geometry
    assert z['group_token_counts'].tolist()==meta['group_token_counts']
    assert str(z['protocol'])=='target-token-region-stats-v2'
    assert z['target_query_count'].item()==meta['target_query_count']==1024
    assert np.array_equal(z['layer_ids'],np.arange(36)) and z['layer_ids'].dtype==np.int64
    assert z['timesteps'].shape==(30,) and z['timesteps'].dtype==np.float32 and np.isfinite(z['timesteps']).all() and np.all(np.diff(z['timesteps'])<0)
    assert z['group_stats'].shape==(30,36,16,len(names),len(z['stat_names']))
    assert z['text_mean'].shape==tuple(meta['text_shape']) and z['text_mean'].shape[:3]==(30,36,16)
    assert z['region_mean'].shape==(30,36,16,t,2,8,8)==tuple(meta['region_shape'])
    assert meta['shape']==[30,36,16,len(names)]
    assert set(z['stat_names'].tolist())=={'mean','std','p10','p50','p90'}
    for key in ['group_stats','text_mean','region_mean']:
        arr=z[key];assert arr.dtype==np.float32 and np.isfinite(arr).all() and np.all(arr>=0),key
    stat=z['group_stats'];sn=z['stat_names'].tolist()
    assert np.all(stat[...,sn.index('p10')]<=stat[...,sn.index('p50')]) and np.all(stat[...,sn.index('p50')]<=stat[...,sn.index('p90')])
    return names,geometry

def density(base,mass,count,N):
    return dict(base,mass=float(mass),token_count=int(count),total_key_tokens=int(N),present=bool(count),
        per_token_mass=float(mass/count) if count else None,enrichment=float(mass/(count/N)) if count else None,
        density_null_reason=None if count else 'structural_zero')

def session_worker(args):
    items,temp=args;sid=items[0]['session_id'];out={k:[] for k in TABLES};audits=[]
    for item in items:
        t=item['generation_turn'];path=pathlib.Path(item['original_path']);remote=item['host']=='a800_1'
        if remote:
            path=pathlib.Path(temp)/f'{sid.replace("/","__")}_turn_{t}.attention.npz'
            download(f'/attention/{sid}/turn_{t}.attention.npz',path,item['bytes'],item['sha256'])
        try:
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],('source receipt',sid,t)
            j=json.loads(pathlib.Path(item['json_path']).read_text());base=dict(session_id=sid,generation_turn=t,source_id=item['source_id'])
            with np.load(path,allow_pickle=False) as z:
                names,geometry=layout_audit(z,j);counts=z['group_token_counts'];N=int(counts.sum())
                mass=z['group_stats'][...,z['stat_names'].tolist().index('mean')].astype(np.float64).reshape(17280,len(names))
                means=mass.mean(0);conservation=float(np.abs(mass.sum(1)-1).max())
                json_error=max(abs(means[i]-j['backend']['attention']['mean_mass'][g]) for i,g in enumerate(names))
                assert conservation<=1e-3 and json_error<=5e-5
                rows={g:density(dict(base,group_name=g,view='exclusive_raw'),means[i],counts[i],N) for i,g in enumerate(names)}
                out['group_turn'].extend(rows.values())
                markers=z['image_marker_mean'].astype(np.float64).reshape(17280,t,2,2)
                for im in range(t):
                    for mi,m in enumerate(['vit','vae']):
                        spatial=mass[:,names.index(f'I{im}_{m}')];block=spatial+markers[:,im,mi,:].sum(1);valid=block>0
                        for k,label in enumerate(['start','end']):
                            mm=markers[:,im,mi,k]
                            out['image_marker_turn'].append(dict(base,image_index=im,modality=m,marker=label,object_role='original_image' if im==0 else 'history_image',
                                mean_raw_mass=float(mm.mean()),mean_block_share=float((mm[valid]/block[valid]).mean()) if valid.any() else None,
                                n_positions_positive=int(valid.sum()),mean_spatial_mass=float(spatial.mean())))

                # Same-round same-type pairs: ratios computed before averaging.
                pairs=[('text',a,b,'text') for a,b in [(1,2),(1,3),(2,3)] if b<=t]
                pairs += [('image',a,b,m) for a,b in [(0,1),(0,2),(1,2)] if b<t for m in ['vit','vae','combined']]
                for typ,a,b,m in pairs:
                    ga=[f'T{a}'] if typ=='text' else [f'I{a}_{k}' for k in (['vit','vae'] if m=='combined' else [m])]
                    gb=[f'T{b}'] if typ=='text' else [f'I{b}_{k}' for k in (['vit','vae'] if m=='combined' else [m])]
                    ia=[names.index(g) for g in ga];ib=[names.index(g) for g in gb]
                    aa=mass[:,ia].sum(1);bb=mass[:,ib].sum(1);total=aa+bb;valid=total>0;sens=total>1e-6
                    na=counts[ia].sum();nb=counts[ib].sum();da=aa/na;db=bb/nb
                    ratio=np.divide(aa,total,out=np.zeros_like(aa),where=valid)
                    dratio=np.divide(da,da+db,out=np.zeros_like(aa),where=valid)
                    out['within_type_pairs'].append(dict(base,object_type=typ,modality=m,older_index=a,newer_index=b,
                        older_share=float(ratio[valid].mean()) if valid.any() else None,
                        older_density_share=float(dratio[valid].mean()) if valid.any() else None,
                        older_share_sensitivity=float(ratio[sens].mean()) if sens.any() else None,
                        pooled_older_share=float(aa.mean()/total.mean()) if valid.any() else None,
                        positive_positions=int(valid.sum()),sensitivity_positions=int(sens.sum()),n_positions_total=17280,
                        zero_mass_fraction=float((~valid).mean()),older_token_count=int(na),newer_token_count=int(nb)))

                for i in range(t):
                    pair=[rows[f'I{i}_{m}'] for m in ['vit','vae']]
                    out['image_combined_turn'].append(density(dict(base,image_index=i,group_name=f'I{i}_combined',view='additional_nonexclusive'),sum(x['mass'] for x in pair),sum(x['token_count'] for x in pair),N))
                mapped={
                    'current_instruction':[f'T{t}'],'history_instructions':[f'T{k}' for k in range(1,t)],
                    'original_image':['I0_vit','I0_vae'],'history_images':[f'I{i}_{m}' for i in range(1,t) for m in ['vit','vae']],
                    'target_image':['target_image'],'context_other':['context_other'],'generation_markers':['generation_markers']}
                cats=[]
                for cat,gs in mapped.items():
                    cats.append(density(dict(base,category=cat,view='exclusive_category'),sum(rows[g]['mass'] for g in gs),sum(rows[g]['token_count'] for g in gs),N))
                assert abs(sum(x['mass'] for x in cats)-means.sum())<=1e-12 and sum(x['token_count'] for x in cats)==N
                out['category_turn'].extend(cats)
                for cat,ims in [('original_image',[0]),('history_images',list(range(1,t)))]:
                    for m in ['vit','vae']:
                        gs=[f'I{i}_{m}' for i in ims]
                        out['category_modality_turn'].append(density(dict(base,category=cat,modality=m,view='additional_nonexclusive'),sum(rows[g]['mass'] for g in gs),sum(rows[g]['token_count'] for g in gs),N))
                for g,row in rows.items():
                    if not g.startswith('T') and not g.startswith('I'):continue
                    idx=int(g[1:].split('_')[0]);kind='text' if g.startswith('T') else 'image';modality='text' if kind=='text' else g.split('_')[1]
                    identity=j['backend']['image_hashes'][idx] if kind=='image' else next(x['text'] for x in j['backend']['positive_trace'] if x.get('role') in ['history','current'] and x['turn']==idx)
                    out['object_trajectories'].append(dict(row,object_id=f'{sid}/{kind}/{idx}/{modality}',object_index=idx,object_type=kind,modality=modality,
                        object_role=('current_instruction' if idx==t else 'history_instruction') if kind=='text' else ('original_image' if idx==0 else 'history_image'),
                        age=t-idx if kind=='text' else t-1-idx,object_identity=identity))
                for row in out['image_combined_turn']:
                    if row['generation_turn']!=t:continue
                    idx=row['image_index'];out['object_trajectories'].append(dict(row,object_id=f'{sid}/image/{idx}/combined',object_index=idx,object_type='image',modality='combined',
                        object_role='original_image' if idx==0 else 'history_image',age=t-1-idx,object_identity=j['backend']['image_hashes'][idx]))
                support_results=[];support_ids=[];Ns=[];arr_valid=[];arr_low=[];arr_positive=[];array_error=0.;scalar_error=0.;support_error=0.
                supports=[]
                texts=z['text_token_turns'];tm=z['text_mean'].astype(np.float64).reshape(17280,-1)
                regions=z['region_mean'].astype(np.float64).reshape(17280,t,2,64);rc=z['region_token_counts'].reshape(t,2,64)
                for idx in range(1,t+1):
                    select=texts==idx;supports.append((f'T{idx}','text',idx,'text',tm[:,select],select))
                for idx in range(t):
                    for mi,m in enumerate(['vit','vae']):supports.append((f'I{idx}_{m}','image',idx,m,regions[:,idx,mi,rc[idx,mi]>0],mi))
                for support_id,kind,idx,m,x,extra in supports:
                    result=calculate(x);group_idx=names.index(support_id)
                    error=float(np.abs(result['values'][:,0]-mass[:,group_idx]).max());support_error=max(support_error,error);assert error<=1e-3
                    role=('current_instruction' if idx==t else 'history_instruction') if kind=='text' else ('original_image' if idx==0 else 'history_image')
                    sb=dict(base,support_id=support_id,support_type=kind,object_index=idx,modality=m,object_role=role,age=t-idx if kind=='text' else t-1-idx,
                        object_id=f'{sid}/{kind}/{idx}/{m}',conditioning='positive_support_mass',source_group_mass=rows[support_id]['mass'])
                    out['concentration_turn'].append(scalar_rows(result,sb))
                    stored=result['values'].astype(np.float32)
                    array_error=max(array_error,float(np.abs(stored.astype(np.float64)-result['values']).max()))
                    for c in range(len(METRICS)):
                        vm=result['valid'][:,c]
                        if vm.any():scalar_error=max(scalar_error,abs(stored[vm,c].astype(np.float64).mean()-result['values'][vm,c].mean()))
                    support_results.append(stored.reshape(30,36,16,len(METRICS)));support_ids.append(support_id);Ns.append(result['N'])
                    arr_valid.append(result['valid'].reshape(30,36,16,len(METRICS)));arr_positive.append(result['positive'].reshape(30,36,16));arr_low.append(result['low'].reshape(30,36,16))
                    def profile(k):
                        return dict(mean_raw_mass=float(result['raw'][k]),mean_conditional_share=float(result['conditional'][k]) if result['conditional'] is not None else None,
                            pooled_conditional_share=float(result['pooled'][k]) if result['pooled'] is not None else None,n_positions_total=17280,n_positions_conditional=int(result['positive'].sum()),
                            conditional_null_reason=None if result['positive'].any() else 'zero_support_mass')
                    if kind=='text':
                        trace=next(a for a in j['backend']['positive_trace'] if a.get('role') in ['history','current'] and a['turn']==idx)
                        selected=np.flatnonzero(extra)
                        for k,key in enumerate(selected):
                            start,end=map(int,z['text_token_offsets'][key]);piece=trace['text'][start:end]
                            out['text_token_profiles'].append(dict(sb,token_ordinal=k,token_id=int(z['text_token_ids'][key]),key_position=int(z['text_key_positions'][key]),
                                offset_start=start,offset_end=end,instruction_text=trace['text'],token_piece=piece,token_piece_method='original character offsets; overlapping spans retained',**profile(k)))
                    else:
                        geom=next(g for g in geometry if g['kind']==m and g['image_index']==idx);valid_indices=np.flatnonzero(rc[idx,extra]>0);mapping={cell:k for k,cell in enumerate(valid_indices)}
                        for cell in range(64):
                            n=int(rc[idx,extra,cell]);prof=profile(mapping[cell]) if n else dict(mean_raw_mass=0.,mean_conditional_share=None,pooled_conditional_share=None,n_positions_total=17280,n_positions_conditional=0,conditional_null_reason='empty_region')
                            out['image_region_profiles'].append(dict(sb,row=cell//8,col=cell%8,region_token_count=n,image_hash=j['backend']['image_hashes'][idx],geometry_json=json.dumps(geom,ensure_ascii=False),
                                region_present=bool(n),per_token_raw_mass=prof['mean_raw_mass']/n if n else None,density_null_reason=None if n else 'empty_region',**prof))
                # Invalid metric values are zero placeholders, never observations.
                ap=ROOT/'arrays'/sid/f'turn_{t}.npz';ap.parent.mkdir(parents=True,exist_ok=True)
                part=ap.with_name(ap.name+'.part')
                with open(part,'wb') as f:np.savez_compressed(f,metric_values=np.stack(support_results,axis=3),metric_valid=np.stack(arr_valid,axis=3),
                    positive_mass=np.stack(arr_positive,axis=3),low_mass=np.stack(arr_low,axis=3),support_ids=np.array(support_ids),metric_names=np.array(METRICS),
                    support_N=np.array(Ns,dtype=np.int64),timesteps=z['timesteps'],step_ids=np.arange(30),layer_ids=z['layer_ids'],head_ids=np.arange(16),source_id=np.array(item['source_id']),
                    source_sha256=np.array(item['sha256']),calculation_dtype=np.array('float64'),storage_dtype=np.array('float32'))
                os.replace(part,ap)
                audits.append(dict(source_id=item['source_id'],session_id=sid,generation_turn=t,host=item['host'],bytes=item['bytes'],sha256=item['sha256'],
                    max_probability_sum_error=conservation,max_json_mean_error=json_error,max_support_mass_error=support_error,
                    max_array_rounding_error=array_error,max_scalar_storage_error=scalar_error,sha256_verified=True,index_identity_verified=True))
        finally:
            if remote:path.unlink(missing_ok=True)
    dest=pathlib.Path(temp)/(sid.replace('/','__')+'.result.json.gz')
    with gzip.open(dest,'wt',encoding='utf8') as f:json.dump(dict(tables=out,audits=audits),f,ensure_ascii=False,allow_nan=False)
    return str(dest)

def paired(trajectories):
    grouped=defaultdict(dict);out=[]
    for r in trajectories:grouped[r['object_id']][r['generation_turn']]=r
    for oid,rs in sorted(grouped.items()):
        for a,b in [(1,2),(2,3),(1,3)]:
            if a not in rs or b not in rs:continue
            x,y=rs[a],rs[b];assert x['object_identity']==y['object_identity'] and x['token_count']==y['token_count']
            out.append(dict(session_id=x['session_id'],source_id_from=x['source_id'],source_id_to=y['source_id'],object_id=oid,object_type=x['object_type'],object_index=x['object_index'],
                modality=x['modality'],object_role_from=x['object_role'],object_role_to=y['object_role'],turn_from=a,turn_to=b,age_from=x['age'],age_to=y['age'],object_identity=x['object_identity'],
                delta_mass=y['mass']-x['mass'],delta_per_token_mass=y['per_token_mass']-x['per_token_mass'],delta_enrichment=y['enrichment']-x['enrichment']))
    return out

def composition_summary(tables,clusters):
    out=[]
    for table,keys in [('group_turn',['group_name']),('image_combined_turn',['group_name']),('category_turn',['category']),('category_modality_turn',['category','modality'])]:
        grouped=defaultdict(list)
        for r in tables[table]:grouped[tuple(r[k] for k in keys)].append(r)
        for key,rows in sorted(grouped.items()):
            for metric in ['mass','per_token_mass','enrichment']:
                for turn in [1,2,3,'overall']:
                    selected=[r for r in rows if turn=='overall' or r['generation_turn']==turn]
                    if not selected:continue
                    by_session=defaultdict(list)
                    for r in selected:by_session[r['session_id']].append(r)
                    vals={s:equal_average(rs,metric) for s,rs in by_session.items()}
                    zeros=sum(all(not r['present'] for r in rs) for rs in by_session.values())
                    out.append(dict(table=table,**dict(zip(keys,key)),generation_turn=turn,metric=metric,aggregation_level='session_equal_turn' if turn!='overall' else 'session_equal_overall',
                        conditioning='all_turns_with_structural_zero' if metric=='mass' else 'present_only',n_valid_turns=sum(r[metric] is not None for r in selected),
                        **describe(vals,clusters,(table,key,metric,turn),n_structural_zero=zeros)))
    return out

def object_summaries(trajectories,changes,clusters):
    traj=[];ps=[];matrix=[]
    groups=defaultdict(list)
    for r in trajectories:groups[(r['object_type'],r['object_index'],r['modality'],r['generation_turn'])].append(r)
    for key,rows in sorted(groups.items()):
        for metric in ['mass','per_token_mass','enrichment']:
            vals={r['session_id']:r[metric] for r in rows}
            traj.append(dict(object_type=key[0],object_index=key[1],modality=key[2],generation_turn=key[3],metric=metric,aggregation_level='session_equal_turn',conditioning='present_only',
                **describe(vals,clusters,('trajectory',key,metric))))
    for sid in sorted(clusters):
        lookup={(r['object_type'],r['object_index'],r['modality'],r['generation_turn']):r for r in trajectories if r['session_id']==sid}
        for typ,indices,modalities in [('text',range(1,4),['text']),('image',range(3),['vit','vae','combined'])]:
            for idx in indices:
                for m in modalities:
                    for t in [1,2,3]:
                        r=lookup.get((typ,idx,m,t));matrix.append(dict(session_id=sid,source_id=f'{sid}/turn_{t}',generation_turn=t,object_type=typ,object_index=idx,modality=m,
                            mass=r['mass'] if r else None,per_token_mass=r['per_token_mass'] if r else None,enrichment=r['enrichment'] if r else None,
                            present=r is not None,missing_mask=r is None,null_reason=None if r else 'not_yet_present',n_sessions=1 if r else 0))
    pg=defaultdict(list)
    for r in changes:pg[(r['object_type'],r['object_index'],r['modality'],r['turn_from'],r['turn_to'])].append(r)
    for key,rows in sorted(pg.items()):
        for metric in ['delta_mass','delta_per_token_mass','delta_enrichment']:
            vals={r['session_id']:r[metric] for r in rows};x=np.array(list(vals.values()))
            ps.append(dict(object_type=key[0],object_index=key[1],modality=key[2],turn_from=key[3],turn_to=key[4],metric=metric,aggregation_level='session_equal_paired',conditioning='both_present',
                positive_fraction=float((x>1e-12).mean()),negative_fraction=float((x<-1e-12).mean()),near_zero_fraction=float((np.abs(x)<=1e-12).mean()),near_zero_tolerance=1e-12,
                **describe(vals,clusters,('paired',key,metric))))
    return traj,ps,matrix

def concentration_summary(rows,clusters):
    out=[]
    for grouping in ['object_index','object_role']:
        grouped=defaultdict(list)
        for r in rows:grouped[(r['support_type'],r['modality'],r[grouping])].append(r)
        for key,rs in sorted(grouped.items()):
            for metric in METRICS+['zero_mass_fraction','low_mass_fraction','low_mass_fraction_positive']:
                variants=['main','mass_gt_1e-6'] if metric in METRICS else ['main']
                for variant in variants:
                    field=metric if variant=='main' else metric+'_sensitivity'
                    for turn in [1,2,3,'overall']:
                        selected=[r for r in rs if turn=='overall' or r['generation_turn']==turn]
                        if not selected:continue
                        within=defaultdict(list)
                        for r in selected:within[(r['session_id'],r['generation_turn'])].append(r)
                        by_session=defaultdict(list)
                        for (s,t),rr in within.items():by_session[s].append(dict(value=equal_average(rr,field)))
                        vals={s:equal_average(rr) for s,rr in by_session.items()}
                        out.append(dict(grouping=grouping,support_type=key[0],modality=key[1],**{grouping:key[2]},generation_turn=turn,metric=metric,variant=variant,
                            aggregation_level='session_equal_turn' if turn!='overall' else 'session_equal_overall',conditioning='present_only',
                            n_supports_total=len(selected),n_supports_valid=sum(r[field] is not None for r in selected),n_valid_turns=sum(r['value'] is not None for rr in by_session.values() for r in rr),
                            **describe(vals,clusters,('concentration',grouping,key,metric,variant,turn))))
    return out

def within_type_summary(rows,clusters):
    groups=defaultdict(list);out=[]
    for r in rows:groups[(r['object_type'],r['modality'],r['older_index'],r['newer_index'],r['generation_turn'])].append(r)
    for key,rr in groups.items():
        for metric in ['older_share','older_density_share','older_share_sensitivity','pooled_older_share']:
            vals={r['session_id']:r[metric] for r in rr}
            out.append(dict(object_type=key[0],modality=key[1],older_index=key[2],newer_index=key[3],generation_turn=key[4],metric=metric,
                aggregation_level='session_equal_turn',conditioning='both_present_positive_pair_mass',**describe(vals,clusters,('within_type',key,metric))))
    return out

def run(temp,workers):
    items,clusters,samples=prepare();status('computing',1,progress={'sessions':0,'turns':0},workers=workers,issues=[])
    by=defaultdict(list)
    for item in items:by[item['session_id']].append(item)
    futures={};paths=[];done=0;start=time.monotonic();sids=iter(sorted(by))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        def submit_one():
            try:sid=next(sids)
            except StopIteration:return False
            futures[pool.submit(session_worker,(by[sid],temp))]=sid;return True
        for _ in range(workers):submit_one()
        while futures:
            completed,_=concurrent.futures.wait(futures,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in completed:
                sid=futures.pop(f);paths.append(f.result());done+=1;submit_one()
                print(json.dumps(dict(at=now(),sessions=done,turns=done*3,elapsed_seconds=round(time.monotonic()-start,2))),flush=True)
            status('computing',1,progress={'sessions':done,'turns':done*3},workers=workers,elapsed_seconds=round(time.monotonic()-start,2),issues=[])
    tables={k:[] for k in TABLES};audits=[]
    status('aggregating',1,progress={'sessions':done,'turns':done*3},issues=[])
    for p in sorted(paths):
        with gzip.open(p,'rt',encoding='utf8') as f:data=json.load(f)
        audits.extend(data['audits'])
        for k in TABLES:tables[k].extend(data['tables'][k])
    audit_lookup={r['source_id']:r for r in audits}
    for item in items:item.update(audit_lookup[item['source_id']])
    write_gz_jsonl(ROOT/'metadata/source_manifest.jsonl.gz',items)
    atomic_json(ROOT/'validation/input_audit.json',dict(task_id=TASK,status='passed',sessions=572,turns=1716,receipts=572,files_sha256_verified=1716,
        bytes=sum(r['bytes'] for r in audits),host_sessions=dict(Counter(x['host'] for x in [v[0] for v in by.values()])),host_files=dict(Counter(x['host'] for x in items)),
        max_probability_sum_error=max(r['max_probability_sum_error'] for r in audits),max_json_mean_error=max(r['max_json_mean_error'] for r in audits),max_support_mass_error=max(r['max_support_mass_error'] for r in audits),
        max_array_rounding_error=max(r['max_array_rounding_error'] for r in audits),max_scalar_storage_error=max(r['max_scalar_storage_error'] for r in audits),
        thresholds={'probability_sum':1e-3,'support_mass':1e-3,'json_mean':5e-5},identity_checks='all session/protocol/history/text/index/geometry/shape/dtype; receipt-bound size and full SHA256',issues=[]))
    tables['within_type_summary']=within_type_summary(tables['within_type_pairs'],clusters)
    tables['paired_changes']=paired(tables['object_trajectories'])
    tables['composition_summary']=composition_summary(tables,clusters)
    tables['trajectory_summary'],tables['paired_summary'],tables['trajectory_matrix']=object_summaries(tables['object_trajectories'],tables['paired_changes'],clusters)
    tables['concentration_summary']=concentration_summary(tables['concentration_turn'],clusters)
    for name,rows in tables.items():write_csv(name,rows)
    # The support-level table has 10296 rows; metrics and their denominators are wide columns.
    expected={'group_turn':15444,'image_combined_turn':3432,'category_turn':12012,'category_modality_turn':6864,'object_trajectories':13728,
        'paired_changes':9152,'concentration_turn':10296,'trajectory_matrix':20592,'image_region_profiles':439296}
    for name,n in expected.items():assert len(tables[name])==n,(name,len(tables[name]),n)
    atomic_json(ROOT/'logs/computation_complete.json',dict(task_id=TASK,at=now(),workers=workers,elapsed_seconds=time.monotonic()-start,table_rows={k:len(v) for k,v in tables.items()},array_files=1716))
    return tables

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--temp',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    try:run(a.temp,a.workers)
    except Exception as e:
        atomic_json(ROOT/'validation/computation_failure.json',dict(task_id=TASK,status='failed',at=now(),error=str(e),traceback=traceback.format_exc()))
        status('computation_failed',1,issues=[str(e)]);raise

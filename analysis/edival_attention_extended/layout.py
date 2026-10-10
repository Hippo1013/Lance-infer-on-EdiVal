"""Read-only v2 layout audit, copied from accepted semantic analysis.

This supports host-local observations without requiring a copy of old analysis.
"""
import json
import numpy as np

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

#!/usr/bin/env python3
"""Audit actual scoring evidence without loading any GPU model."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lance_mice.scoring.core import (read,write,digest,source_pins,ga_prompt,parse_ga,JUDGES,
                                    SETTINGS,vote_mean,request_identity,sampling_seed)
from lance_mice.scoring.runner import saved,artifact_path

def audit(output):
    m=read(output/'manifest.json');fingerprint=m.pop('fingerprint')
    assert digest(m)==fingerprint and m['sources']==source_pins() and m['settings']==SETTINGS
    m['fingerprint']=fingerprint
    summary=read(output/'summary.json');assert summary['status']=='complete'
    scores=read(output/'scores.json');assert len(scores)==3*len(m['records'])
    assert len({(r['session_id'],r['turn']) for r in scores})==len(scores)
    traces=0;compared=0;missing_if=0
    for row in m['records']:
        for t in range(1,4):
            metric=saved(artifact_path(output,'metrics',row,t),fingerprint);assert metric['status']=='ok'
            for key in metric['detection_requests']:
                d=read(output/'detections'/f'{key}.json')
                assert d['fingerprint']==fingerprint and digest(d['detections'])==d['checksum']
            q=[]
            for name in JUDGES:
                r=saved(artifact_path(output,name,row,t),fingerprint);assert r['status']=='ok'
                votes=list(range(1,SETTINGS['votes_per_judge']+1))
                assert [v['vote'] for v in r['votes']]==votes
                ga=[x for x in r['traces'] if x['kind']=='GA'];assert [x['vote'] for x in ga]==votes
                for call,vote in zip(ga,r['votes']):
                    assert call['image_hashes']==row['pixel_hashes'][:t+1]
                    assert call['prompt']==ga_prompt(row['split'],row['metadata']['instruction'][:t],row['metadata']['formatted_instruction'][:t])
                    assert parse_ga(call['raw'],t)==vote['GA_prefix']['score']
                assert r['GA_prefix']==vote_mean([v['GA_prefix'] for v in r['votes']])
                calls=[x for x in r['traces'] if x['kind']=='IF']
                for call in calls:
                    assert call['image_hashes'] in [row['pixel_hashes'][t-1:t+1],row['pixel_hashes'][t:t+1]]
                if 'shared_if' in metric:
                    assert not calls and r['IF']==metric['shared_if']
                    assert all(v['IF']==metric['shared_if'] for v in r['votes'])
                else:
                    assert [call['vote'] for call in calls]==votes
                    assert r['IF']==vote_mean([v['IF'] for v in r['votes']])
                if r['IF']['status']=='invalid_annotation':missing_if+=1
                for call in r['traces']:
                    assert call['raw'].strip() and call['finish_reason']=='stop' and not call.get('error')
                    assert call['temperature']==SETTINGS['temperature']
                    assert call['sampling_seed']==sampling_seed(call['vote'])
                    assert call['request_hash']==request_identity(call['image_hashes'],call['prompt'],call['vote'])
                    assert call['prompt_tokens']+call['completion_tokens']<=call.get('runtime_context_limit',m.get('context_limits',{}).get(name,8192))
                traces+=len(r['traces']);q.append(calls)
            if q[0] and q[1]:
                assert [x['request_hash'] for x in q[0]]==[x['request_hash'] for x in q[1]];compared+=1
    return {'status':'passed','scope':'engineering evidence, not human accuracy','sessions':len(m['records']),
            'turns':len(scores),'judge_requests':traces,'paired_if_requests_checked':compared,
            'invalid_if_judge_records':missing_if,'protocol':m['protocol'],'votes_per_judge':SETTINGS['votes_per_judge'],
            'checks':['manifest and result checksums','shared pure IF','identical dual-judge requests',
                      'IF previous/current images','GA complete prefix images and exact prompt',
                      'raw response parsing and token limits','detection cache integrity','coverage']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    value=audit(args.output);write(args.output/'validation.json',value);print(value)

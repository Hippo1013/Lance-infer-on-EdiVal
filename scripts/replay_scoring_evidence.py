#!/usr/bin/env python3
"""Narrowly pinned migrations: v3/v4 evidence -> final parser and runtime adapter.

Preserves generated answers and unchanged metric measurements. Does not copy judge
scores: the new runner must re-interpret every raw response and recompute IF/GA.
Only valid complete generations are cached. Missing calls will be newly evaluated.
"""
import argparse
import fcntl
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lance_mice.scoring.core import ROOT,SETTINGS,JUDGES,read,write,digest,sha,source_pins
from lance_mice.scoring.runner import persist
from lance_mice.scoring.request_cache import RequestCache

LEGACY = {
 'src/lance_mice/scoring/core.py':'6799ed590fe47a3f320d5be1230b9254ca1e6a9e9f6fa3e1aad45ab994c28e13',
 'src/lance_mice/scoring/judge.py':'7b1e3b0b3be17758f3a9f8171f5a91d962f1706607127403cbb2ff0f3aebc084',
 'src/lance_mice/scoring/runner.py':'9135747fba967d3161eba5069e9d7c8aaf4da22b38bf68049a8b53166bc7f3ea',
}

V4 = {'src/lance_mice/scoring/core.py': '58f6747a2751847a7b80af73004dceb394860727a8ff89b4884e0b512d1cc415', 'src/lance_mice/scoring/judge.py': 'bbf9ed97f59cdd4b5927d6ca16212edc07ac542d7bcd90211ec16c955440dbd8', 'src/lance_mice/scoring/runner.py': 'f6dfaa26e7ae9b65cecea856605aed4d303f4ffa43b71b1e500e350880f3383e', 'src/lance_mice/scoring/request_cache.py': 'd0adfa72644a212d48d0d3f351a3a903f26b2888b1bb13c302fab0735282af55'}
TARGET = {'src/lance_mice/scoring/core.py': '58f6747a2751847a7b80af73004dceb394860727a8ff89b4884e0b512d1cc415', 'src/lance_mice/scoring/judge.py': 'f631240f719158e045b66d6fca5c85c57742e0c59649775500a1ccda9ebdfc60', 'src/lance_mice/scoring/runner.py': '41e993f97c6a592493a7725074f5aab043ab715ebb0afaf8d7cede61fa8a10a7', 'src/lance_mice/scoring/request_cache.py': 'd0adfa72644a212d48d0d3f351a3a903f26b2888b1bb13c302fab0735282af55'}

def checked(path,fp):
    value=read(path);checksum=value.pop('checksum')
    if digest(value)!=checksum or value['fingerprint']!=fp:raise ValueError('Corrupted source evidence: '+str(path))
    return value

def import_raw(source,output,m,old_fp,judges):
    counts={}
    for judge,model_name in JUDGES.items():
        if judge not in judges:continue
        pin=next(x for x in m['models']['models'] if x['name']==model_name);count=0
        for sample in m['records']:
            for turn in range(1,4):
                p=source/judge/sample['session_id']/f'turn_{turn}.json'
                if not p.exists():continue
                row=checked(p,old_fp)
                for trace in row.get('traces',[]):
                    if trace.get('finish_reason')!='stop' or not trace.get('raw','').strip() or trace.get('error'):continue
                    if Path(trace['model']).name!=model_name:raise ValueError('Judge model identity changed')
                    expected=sample['pixel_hashes'][:turn+1] if trace['kind']=='GA' else None
                    if expected is not None and trace['image_hashes']!=expected:raise ValueError('GA image identity changed')
                    if trace['kind']=='IF' and trace['image_hashes'] not in [sample['pixel_hashes'][turn-1:turn+1],sample['pixel_hashes'][turn:turn+1]]:
                        raise ValueError('IF image identity changed')
                    cache=RequestCache(output/'requests'/judge,trace['model'],pin,old_fp)
                    cache.put(trace,{'kind':'same-session immutable raw-response replay','path':str(p),'sha256':sha(p)})
                    count+=1
        counts[judge]=count
    return counts

def migrate(source,output,dataset,judges):
    if output.exists():raise ValueError('Replay output must be new')
    # Caller must wait for all producer stages; lock parent plus isolated workers.
    import contextlib
    with contextlib.ExitStack() as stack:
        for folder in [source]+[source/('worker_'+j) for j in judges if (source/('worker_'+j)).exists()]:
            lock=stack.enter_context((folder/'.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read(source/'manifest.json');old_fp=m.pop('fingerprint')
        if digest(m)!=old_fp or m['settings']!=SETTINGS:raise ValueError('Changed manifest or model request settings')
        if m['selection']=='all':raise ValueError('This migration is restricted to pre-full-run validation evidence')
        pins=source_pins()
        if any(pins.get(name)!=pin for name,pin in TARGET.items()):raise ValueError('Unsupported target scoring implementation')
        for name,pin in m['sources'].items():
            if name in LEGACY or name in V4:
                if pin not in {LEGACY.get(name),V4.get(name)}:raise ValueError('Unsupported producer implementation: '+name)
            elif pins.get(name)!=pin:raise ValueError('A metric, prompt, or inference rule changed: '+name)
        if m['models']!=read(ROOT/'configs/scoring_models.json'):raise ValueError('Model revisions changed')
        for split,pin in m['dataset_hashes'].items():
            if sha(dataset/split/'test_metadata.jsonl')!=pin:raise ValueError('Metadata changed')
        for row in m['records']:
            if [sha(p) for p in row['images']]!=row['image_sha256']:raise ValueError('Image changed')
        inference=Path(m['records'][0]['images'][0]).parents[3]
        preflight=read(source/'preflight.json')
        if sha(inference/'validation.json')!=preflight['inference_validation_sha256']:raise ValueError('Original inference audit changed')
        for name,pin in read(inference/'source_hashes.json').items():
            if sha(ROOT/name)!=pin:raise ValueError('Original inference code changed')
        m.update(sources=pins,replay_provenance={'source_manifest':str(source/'manifest.json'),
            'source_manifest_sha256':sha(source/'manifest.json'),'source_fingerprint':old_fp,
            'scope':'Reverified selected inputs; unchanged metrics; raw responses only, no judge scores'})
        new_fp=digest(m);write(output/'manifest.json',{**m,'fingerprint':new_fp})
        write(output/'preflight.json',{**preflight,'inherited_full_input_audit':str(source/'preflight.json'),
            'inherited_audit_sha256':sha(source/'preflight.json'),'selected_input_hashes_reverified':True})
        detections=0
        for p in sorted((source/'detections').glob('*.json')):
            row=read(p)
            if row['fingerprint']!=old_fp or digest(row['detections'])!=row['checksum']:raise ValueError('Detection cache changed')
            row.update(fingerprint=new_fp,reused_from={'path':str(p),'sha256':sha(p),'fingerprint':old_fp})
            write(output/'detections'/p.name,row);detections+=1
        metrics=0
        for sample in m['records']:
            for turn in range(1,4):
                p=source/'metrics'/sample['session_id']/f'turn_{turn}.json';row=checked(p,old_fp)
                if row['status']!='ok':raise ValueError('Metrics are incomplete')
                row.update(fingerprint=new_fp,reused_from={'path':str(p),'sha256':sha(p),'fingerprint':old_fp})
                persist(output/'metrics'/sample['session_id']/p.name,row);metrics+=1
        write(output/'status_metrics.json',{'status':'complete','method':'Reuse of unchanged verified metrics','source':str(source)})
        counts=import_raw(source,output,m,old_fp,judges)
        value={'status':'passed','source':str(source),'source_fingerprint':old_fp,'target_fingerprint':new_fp,
               'metrics_reused':metrics,'detections_reused':detections,'raw_response_counts':counts,
               'judge_scores_copied':False,'fresh_model_calls':'Only cache misses, under unchanged model/input/sampling settings'}
        write(output/'replay_import.json',value);print(value)


def append_judge(source,output,judge):
    import contextlib
    with contextlib.ExitStack() as stack:
        producer=source/('worker_'+judge)
        if not producer.exists():producer=source
        consumer=output/('worker_'+judge)
        if not consumer.exists():consumer=output
        for folder in [producer,consumer]:
            lock=stack.enter_context((folder/'.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(source/'manifest.json');old_fp=old.pop('fingerprint')
        new=read(output/'manifest.json');new_fp=new.pop('fingerprint')
        if digest(old)!=old_fp or digest(new)!=new_fp:raise ValueError('Manifest corruption')
        if new['sources']!=source_pins() or any(new['sources'].get(k)!=v for k,v in TARGET.items()):raise ValueError('Unsupported target')
        if not any(all(old['sources'].get(k)==v for k,v in version.items()) for version in [LEGACY,V4]):raise ValueError('Unsupported producer')
        if new['replay_provenance']['source_fingerprint']!=old_fp:raise ValueError('Wrong producer')
        for field in ['settings','models','records','dataset_hashes']:
            if old[field]!=new[field]:raise ValueError('Changed request identity')
        if new['settings']!=SETTINGS:raise ValueError('Changed settings')
        status=read(producer/('status_judge_'+judge+'.json'))
        if status['status'] not in ['complete','error']:raise ValueError('Producer is still running')
        counts=import_raw(source,output,new,old_fp,[judge])
        value={'status':'passed','source':str(source),'source_fingerprint':old_fp,'raw_response_counts':counts,'judge_scores_copied':False}
        write(output/('replay_import_'+judge+'.json'),value);print(value)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('/home/chs/dataset/MICE-Bench'));p.add_argument('--only-judge',choices=list(JUDGES));p.add_argument('--append',action='store_true');a=p.parse_args()
    if a.append:
        if not a.only_judge:p.error('--append requires --only-judge')
        append_judge(a.source.resolve(),a.output.resolve(),a.only_judge)
    else:migrate(a.source.resolve(),a.output.resolve(),a.dataset.resolve(),[a.only_judge] if a.only_judge else list(JUDGES))

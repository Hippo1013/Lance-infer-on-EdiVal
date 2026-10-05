"""Prepare -> metrics -> Qwen -> Gemma -> report; explicit full-run boundary."""
from __future__ import annotations
import argparse
from collections import Counter
import html
import io
import base64
import json
from pathlib import Path
import time
from PIL import Image, PngImagePlugin
from lance_mice.images import image_hash
from .core import (ROOT, PROTOCOL, SETTINGS, JUDGES, PURE_IF, TASKS, digest,sha,read,write,source_pins,
                   result,valid_if,ga_prompt,parse_ga,pair_mean,cumulative,mean_report,vote_mean)
from .if_rules import evaluate_single

PngImagePlugin.MAX_TEXT_CHUNK=4*1024*1024

def open_image(path):
    with Image.open(path) as im:return im.convert('RGB')

def select_sessions(rows, selection):
    ranked=sorted(rows,key=lambda r:digest(['mice-score-selection-v1',r['session_id']]))
    if selection=='all':return sorted(rows,key=lambda r:r['session_id'])
    if selection=='calibration':
        return sorted([r for s in ('cm','cu') for r in [v for v in ranked if v['split']==s][:20]],key=lambda r:r['session_id'])
    chosen=[];uncovered=set(TASKS)
    while uncovered:
        candidate=max(ranked,key=lambda r:len(set(r['metadata']['task_type'])&uncovered))
        if not (set(candidate['metadata']['task_type'])&uncovered): raise ValueError('Task coverage impossible')
        chosen.append(candidate);ranked.remove(candidate);uncovered-=set(candidate['metadata']['task_type'])
    # Exercise both splits, invalid annotation and empty CC metadata explicitly.
    predicates=[lambda r,s=s:r['split']==s for s in ('cm','cu')]
    predicates += [lambda r:any(not valid_if(t,f) for t,f in zip(r['metadata']['task_type'],r['metadata']['formatted_instruction'])),
                   lambda r:any(not x for x in r['metadata']['unchanged_objects'])]
    for predicate in predicates:
        if not any(predicate(r) for r in chosen):chosen.append(next(r for r in ranked if predicate(r)))
    return sorted(chosen,key=lambda r:r['session_id'])

def prepare(args):
    if (args.output/'manifest.json').exists():
        raise ValueError('Output already prepared. Use subsequent stages or a new directory; never overwrite a manifest.')
    validation=read(args.inference/'validation.json')
    if validation['status']!='passed' or validation['turns']!=2160:raise ValueError('Full inference audit must pass')
    for name,pin in read(args.inference/'source_hashes.json').items():
        if sha(ROOT/name)!=pin:raise ValueError('Original inference source drift: '+name)
    rows=[];issues=[];pins={};task_counts=Counter()
    for split in ('cm','cu'):
        path=args.dataset/split/'test_metadata.jsonl';pins[split]=sha(path)
        for line in path.read_text().splitlines():
            if not line.strip():continue
            meta=json.loads(line);sid=f'{split}/{Path(meta["image"]).stem}'
            for key in ['instruction','task_type','formatted_instruction','bg_consistency','unchanged_objects','all_objects']:
                if len(meta[key])!=3:raise ValueError('Invalid metadata length: '+sid+'/'+key)
            source=(args.dataset/meta['image']).resolve()
            if not source.is_relative_to(args.dataset.resolve()):raise ValueError('Invalid source path')
            folder=args.inference/'run'/sid;spec=read(folder/'session.json')
            if spec['instructions']!=meta['instruction'] or spec['session_id']!=sid:raise ValueError('Inference identity mismatch')
            images=[folder/'turn_0_input.png']+[folder/f'turn_{i}.png' for i in range(1,4)]
            ph=[image_hash(open_image(p)) for p in images]
            if ph[0]!=image_hash(open_image(source)) or ph[0]!=spec['source_hash']:raise ValueError('Source image mismatch')
            for i in range(3):
                tr=read(folder/f'turn_{i+1}.json')
                if tr['output_hash']!=ph[i+1] or tr['instruction']!=meta['instruction'][i]:raise ValueError('Output image/instruction mismatch')
                task_counts[meta['task_type'][i]]+=1
                if not valid_if(meta['task_type'][i],meta['formatted_instruction'][i]):
                    issues.append({'session_id':sid,'turn':i+1,'kind':'invalid_if_annotation','formatted_instruction':meta['formatted_instruction'][i]})
            rows.append({'session_id':sid,'split':split,'metadata':meta,'images':[str(p) for p in images],
                         'image_sha256':[sha(p) for p in images],'pixel_hashes':ph})
    if len(rows)!=720 or len({r['session_id'] for r in rows})!=720:raise ValueError('Dataset coverage mismatch')
    if args.selection=='all' and not args.allow_full:raise ValueError('Full preparation requires --allow-full')
    selected=select_sessions(rows,args.selection)
    spec={'protocol':PROTOCOL,'settings':SETTINGS,'models':read(ROOT/'configs/scoring_models.json'),
          'context_limits':{'qwen':16384,'gemma':8192},
          'sources':source_pins(),'dataset_hashes':pins,'selection':args.selection,'records':selected}
    manifest={**spec,'fingerprint':digest(spec)}
    write(args.output/'manifest.json',manifest)
    write(args.output/'preflight.json',{'status':'passed','scope':'CPU input audit only','sessions':720,'turns':2160,
         'selected_sessions':len(selected),'selected_turns':3*len(selected),'task_counts':dict(task_counts),'issues':issues,
         'empty_unchanged_turns':sum(not x for r in rows for x in r['metadata']['unchanged_objects']),
         'selection_rule':'Fixed metadata-only hash order; smoke task/edge coverage; calibration first 20 per split',
         'inference_validation_sha256':sha(args.inference/'validation.json')})
    print(json.dumps({'prepared':len(selected),'turns':3*len(selected),'issues':len(issues)}),flush=True)

def load_manifest(args):
    m=read(args.output/'manifest.json');fingerprint=m.pop('fingerprint')
    if digest(m)!=fingerprint:raise ValueError('Manifest checksum mismatch')
    m['fingerprint']=fingerprint
    if m['sources']!=source_pins() or m['settings']!=SETTINGS:raise ValueError('Scoring code/settings changed; prepare a new output')
    if m['selection']=='all' and not args.allow_full:raise ValueError('Full scoring requires --allow-full')
    for split,pin in m['dataset_hashes'].items():
        if sha(args.dataset/split/'test_metadata.jsonl')!=pin:raise ValueError('Dataset changed')
    for r in m['records']:
        if [sha(p) for p in r['images']]!=r['image_sha256']:raise ValueError('Scoring input changed: '+r['session_id'])
    return m

def artifact_path(output,stage,row,turn):return output/stage/row['session_id']/f'turn_{turn}.json'

def saved(path, fingerprint, retry_errors=False):
    if not path.exists():return None
    x=read(path);checksum=x.pop('checksum')
    if digest(x)!=checksum or x['fingerprint']!=fingerprint:raise ValueError('Saved result identity mismatch: '+str(path))
    x['checksum']=checksum
    if x['status']=='error':
        if retry_errors:return None
        raise ValueError('Unresolved saved error; inspect it, then explicitly use --retry-errors')
    return x

def persist(path,row):
    if path.exists():
        prior=read(path)
        checksum=prior.pop('checksum')
        if digest(prior)!=checksum:raise ValueError('Cannot overwrite corrupted evidence')
        if prior.get('status')=='error':
            history=prior.pop('previous_attempts',[])
            row={**row,'previous_attempts':history+[prior]}
    write(path,{**row,'checksum':digest(row)})

def infer_if(row,turn,images,detect,one,two):
    meta=row['metadata'];task=meta['task_type'][turn-1];formatted=meta['formatted_instruction'][turn-1]
    if not valid_if(task,formatted):
        return result('invalid_annotation',reason='Official formatted instruction cannot be parsed',upstream_invalid_format_value=0)
    value,reason=evaluate_single(images[turn-1],images[turn],meta['instruction'][turn-1],formatted,task,detect,one,two)
    return result('ok',float(value),reason=reason,task_type=task)

def metrics_stage(args,m):
    import torch
    from .metrics import Detector,Consistency
    torch.set_num_threads(8);torch.manual_seed(42)
    detector=Detector(args.model_root,args.output/'detections',m['fingerprint']);cc=Consistency(args.model_root,detector)
    for row in m['records']:
        images=[open_image(p) for p in row['images']]
        for t in range(1,4):
            path=artifact_path(args.output,'metrics',row,t)
            if saved(path,m['fingerprint'],args.retry_errors):continue
            out={'fingerprint':m['fingerprint'],'status':'ok','session_id':row['session_id'],'turn':t}
            detector.used={}
            try:
                # Optimistic VLM placeholders enumerate every reachable detection dependency.
                # They are never stored or reported as IF scores.
                planned=infer_if(row,t,images,detector,lambda *a:'yes',lambda *a:'yes')
                task=row['metadata']['task_type'][t-1]
                if task in PURE_IF or planned['status']!='ok':out['shared_if']=planned
                meta={k:row['metadata'][k][t-1] for k in ['unchanged_objects','all_objects','bg_consistency']}
                out['CC']=cc(images[0],images[t],meta)
                out['detection_requests']=detector.used
            except Exception as exc:
                out.update(status='error',error=f'{type(exc).__name__}: {exc}');persist(path,out);raise
            persist(path,out);print('metrics',row['session_id'],t,flush=True)

def judge_stage(args,m):
    from .metrics import Detector
    from .judge import Judge
    from .request_cache import RequestCache
    name=args.judge
    if not name:raise ValueError('--judge required')
    # Fail before GPU model loading if the shared stage is incomplete or corrupted.
    for row in m['records']:
        for t in range(1,4):
            if saved(artifact_path(args.output,'metrics',row,t),m['fingerprint']) is None:raise ValueError('Missing metrics stage')
    detector=Detector(args.model_root,args.output/'detections',m['fingerprint'],readonly=True)
    judge=None
    for row in m['records']:
        images=[open_image(p) for p in row['images']]
        for t in range(1,4):
            path=artifact_path(args.output,name,row,t)
            if saved(path,m['fingerprint'],args.retry_errors):continue
            if judge is None:
                pin=next(x for x in m['models']['models'] if x['name']==JUDGES[name])
                cache=RequestCache(args.output/'requests'/name,args.model_root/JUDGES[name],pin,m['fingerprint'])
                judge=Judge(args.model_root/JUDGES[name],cache,
                            context_limit=m.get('context_limits',{}).get(name,SETTINGS['max_model_len']))
            judge.traces=[];out={'fingerprint':m['fingerprint'],'status':'ok','session_id':row['session_id'],'turn':t,'judge':name,'votes':[]}
            try:
                shared=saved(artifact_path(args.output,'metrics',row,t),m['fingerprint'])
                prompt=ga_prompt(row['split'],row['metadata']['instruction'][:t],row['metadata']['formatted_instruction'][:t])
                for vote in range(1,SETTINGS['votes_per_judge']+1):
                    judge.vote=vote
                    item={'vote':vote};out['votes'].append(item)
                    item['IF']=shared.get('shared_if')
                    if item['IF'] is None:item['IF']=infer_if(row,t,images,detector,judge.one,judge.two)
                    raw=judge.ask(images[:t+1],prompt,'GA')
                    item['GA_prefix']=result('ok',parse_ga(raw,t))
                out['IF']=shared['shared_if'] if 'shared_if' in shared else vote_mean([v['IF'] for v in out['votes']])
                out['GA_prefix']=vote_mean([v['GA_prefix'] for v in out['votes']])
            except Exception as exc:
                out.update(status='error',error=f'{type(exc).__name__}: {exc}',traces=judge.traces);persist(path,out);raise
            out['traces']=judge.traces;persist(path,out);print(name,row['session_id'],t,flush=True)

def report(args,m):
    rows=[];disagreements=[];nonmonotonic=[];errors=[]
    for r in m['records']:
        judges={}
        for name in JUDGES:
            judges[name]=[]
            for t in range(1,4):
                try:x=saved(artifact_path(args.output,name,r,t),m['fingerprint'])
                except ValueError as exc:errors.append(str(exc));x=None
                if x is None:
                    errors.append(f'Missing {name}/{r["session_id"]}/{t}')
                    x={'IF':result('unavailable'),'GA_prefix':result('unavailable'),'traces':[]}
                judges[name].append(x)
        ga={name:cumulative([x['GA_prefix'] for x in judges[name]]) for name in JUDGES}
        for t in range(1,4):
            shared=saved(artifact_path(args.output,'metrics',r,t),m['fingerprint'])
            if shared is None:raise ValueError('Missing metrics result')
            out={'session_id':r['session_id'],'split':r['split'],'turn':t,'task_type':r['metadata']['task_type'][t-1],
                 'instruction':r['metadata']['instruction'][t-1],'CC':shared['CC']}
            for name in JUDGES:
                source=judges[name][t-1];out['IF_'+name]=source['IF'];out['GA_'+name]=ga[name][t-1];out['GA_prefix_'+name]=source['GA_prefix']
                out['IF_votes_'+name]=[v['IF'] for v in source.get('votes',[])]
                out['GA_prefix_votes_'+name]=[v['GA_prefix'] for v in source.get('votes',[])]
                if out['GA_'+name]['status']==source['GA_prefix']['status']=='ok' and out['GA_'+name]['score']<source['GA_prefix']['score']:
                    nonmonotonic.append([r['session_id'],t,name])
            for metric in ('IF','GA','GA_prefix'):
                a,b=out[metric+'_qwen'],out[metric+'_gemma'];out[metric+'_mean']=pair_mean(a,b)
                if metric!='GA_prefix' and a['status']==b['status']=='ok' and a['score']!=b['score']:disagreements.append([r['session_id'],t,metric])
            rows.append(out)
    groups={}
    for split in ('all','cm','cu'):
        for turn in ('all',1,2,3):
            subset=[r for r in rows if (split=='all' or r['split']==split) and (turn=='all' or r['turn']==turn)]
            groups[f'{split}/turn_{turn}']={key:mean_report([r[key] for r in subset]) for key in ['IF_qwen','IF_gemma','IF_mean','CC','GA_qwen','GA_gemma','GA_mean','GA_prefix_qwen','GA_prefix_gemma','GA_prefix_mean']}
    summary={'status':'complete' if not errors else 'incomplete','scope':m['selection'],'protocol':PROTOCOL,'fingerprint':m['fingerprint'],
             'sessions':len(m['records']),'turns':len(rows),'groups':groups,'disagreements':disagreements,
             'nonmonotonic_prefix_judgments':nonmonotonic,'errors':errors,'human_calibration':'pending; review.html and human_review.jsonl',
             'claim':'Engineering validation only. Dual smaller judges are not the official paper judge.'}
    write(args.output/'scores.json',rows);write(args.output/'summary.json',summary)
    build_review(args.output,m,rows,judges=None)
    print(json.dumps({k:summary[k] for k in ['status','sessions','turns','disagreements']}),flush=True)
    if errors:raise ValueError('Incomplete report')

def build_review(output,m,rows,judges=None):
    body=['<!doctype html><meta charset="utf-8"><title>MICE 裁判核对</title><style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px}img{max-width:24%;vertical-align:top}pre{white-space:pre-wrap}section{border-top:2px solid #888;padding:16px 0}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>MICE 小样本裁判核对</h1><p>实验工程核对材料。未作人工准确率校准。原图、生成结果及两位裁判原始回答如下；null 为缺项，不代表编辑失败。</p>']
    template=[]
    for r in m['records']:
        body.append('<section><h2>'+html.escape(r['session_id'])+'</h2><p>从左至右：原图、第一轮、第二轮、第三轮。显示缩略图，评分使用原始分辨率。</p>')
        for p in r['images']:
            im=open_image(p);im.thumbnail((600,600));data=io.BytesIO();im.save(data,format='JPEG',quality=90)
            body.append('<img src="data:image/jpeg;base64,'+base64.b64encode(data.getvalue()).decode()+'">')
        for row in [x for x in rows if x['session_id']==r['session_id']]:
            t=row['turn'];body.append('<h3>第 '+str(t)+' 轮</h3><p>'+html.escape(row['instruction'])+'</p>')
            body.append('<pre>'+html.escape(json.dumps({k:v for k,v in row.items() if k.startswith(('IF','GA','CC'))},ensure_ascii=False,indent=2))+'</pre>')
            for name in JUDGES:
                path=artifact_path(output,name,r,t)
                if path.exists():
                    data=read(path);body.append('<details><summary>'+name+' 原始输入与回答</summary><pre>'+html.escape(json.dumps(data.get('traces',[]),ensure_ascii=False,indent=2))+'</pre></details>')
            template.append({'session_id':r['session_id'],'turn':t,'human_IF':None,'human_GA_prefix':None,'CC_observation':None,'notes':'','reviewer':''})
        body.append('</section>')
    (output/'review.html').write_text('\n'.join(body))
    template_path=output/'human_review.jsonl'
    if not template_path.exists():template_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in template))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','metrics','judge','report'])
    ap.add_argument('--dataset',type=Path,default=Path('/home/chs/dataset/MICE-Bench'))
    ap.add_argument('--inference',type=Path,default=ROOT/'outputs/mice/full_20261004_attention')
    ap.add_argument('--model-root',type=Path,default=Path('/home/chs/model'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection',choices=['smoke','calibration','all'],default='smoke')
    ap.add_argument('--judge',choices=list(JUDGES))
    ap.add_argument('--allow-full',action='store_true')
    ap.add_argument('--retry-errors',action='store_true')
    args=ap.parse_args();args.output=args.output.resolve();args.inference=args.inference.resolve()
    if args.output.is_relative_to(args.inference) or args.inference.is_relative_to(args.output):raise ValueError('Keep scoring outside inference outputs')
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={'stage':args.stage,'judge':args.judge,'status':'running','started_at':time.time()}
        status_path=args.output/('status_'+args.stage+('_'+args.judge if args.judge else '')+'.json')
        write(status_path,status)
        try:
            if args.stage=='prepare':prepare(args)
            else:
                m=load_manifest(args)
                if args.stage=='metrics':metrics_stage(args,m)
                elif args.stage=='judge':judge_stage(args,m)
                else:report(args,m)
            status['status']='complete'
        except Exception as exc:
            status.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            status['finished_at']=time.time();write(status_path,status)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Replay fixed original histories, with corrected read-only attention observation.

One GPU / persistent model. Frozen outputs and scores remain untouched. A pilot
requires exact output pixels, observer on/off agreement and untouched text/ViT
channels. Full collection is gated by both host pilot receipts.
"""
import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
import numpy as np
from PIL import Image
from lance_mice.backend import OmniBackend
from lance_mice.images import image_hash
from lance_mice.settings import Settings

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    t=p.with_suffix(p.suffix+'.part');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def run(args):
    original=args.repo/'outputs/sixrun_20261007/edival_chat/run/edival'
    ids=sorted([p for p in original.iterdir() if p.is_dir() and (p/'session.json').exists()],key=lambda p:int(p.name))
    selected=ids[:2] if args.pilot else ids[args.shard::2]
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'selection.json',{'hostname':socket.gethostname(),'sessions':[p.name for p in selected],'shard':args.shard,'pilot':args.pilot})
    if not args.pilot:
        for name in ['pilot_local.json','pilot_peer.json']:
            gate=json.loads((args.gates/name).read_text())
            if gate.get('status')!='passed' or gate.get('format')!='target-token-region-stats-v2':raise ValueError('Both exact pilot gates required')
    settings=Settings(resolution=512,cache_mode='prefix',history_protocol='lance-history-chat-v1',attention_format='target-token-region-stats-v2')
    runtime=Path(os.environ['PYTHONPATH'].split(':')[0]).parent
    runtime_hash=sha(runtime/'recollection_manifest.json')
    runtime_sources={name:sha(runtime/'src/lance_mice'/name) for name in ['attention.py','attention_regions_v2.py','cache_semantics.py','semantic_pipeline.py','omni_pipeline.py','settings.py']}
    backend=None; started=time.time(); records=[]
    write(args.output/'progress.json',{'status':'loading','started':started,'expected_sessions':len(selected)})
    try:
        backend=OmniBackend(Path('/home/chs/model/Lance'),settings,audit=True,pipeline_class='lance_mice.semantic_pipeline.SemanticLancePipeline')
        for folder in selected:
            session=json.loads((folder/'session.json').read_text());sid=session['session_id'];instructions=session['instructions']
            for mode in (['observed','unobserved'] if args.pilot else ['observed']):
                d=args.output/'run'/folder.name/mode;d.mkdir(parents=True)
                for turn in range(1,4):
                    source=json.loads((folder/f'turn_{turn}.json').read_text())
                    image_paths=[folder/'turn_0_input.png']+[folder/f'turn_{i}.png' for i in range(1,turn)]
                    images=[Image.open(p).convert('RGB') for p in image_paths]
                    if [image_hash(x) for x in images]!=source['input_hashes']:raise ValueError('Original history pixel identity mismatch')
                    attn=d/f'turn_{turn}.attention.npz' if mode=='observed' else None
                    t=time.time();image,meta=backend.edit(sid,images,instructions[:turn],end_session=turn==3,attention_path=attn)
                    output_hash=image_hash(image)
                    if output_hash!=source['output_hash']:raise ValueError(f'Output pixel reproduction failure {sid} turn {turn}: {output_hash} != {source["output_hash"]}')
                    if meta['image_hashes']!=source['input_hashes'] or meta['seed']!=source['backend']['seed']:raise ValueError('History or seed differs')
                    if meta['cache_identity_audit']['status']!='passed':raise ValueError('Real cache audit missing')
                    if mode=='observed':
                        with np.load(attn,allow_pickle=False) as z:
                            if str(z['protocol'])!='target-token-region-stats-v2':raise ValueError('Incorrect observer version')
                            if z['text_mean'].shape[-1]!=len(z['text_key_positions']):raise ValueError('Text channel contamination')
                        if args.pilot:
                            with np.load(folder/f'turn_{turn}.attention.npz',allow_pickle=False) as old,np.load(attn,allow_pickle=False) as new:
                                for key in ('text_mean',):
                                    if not np.array_equal(old[key],new[key]):raise ValueError('Untouched attention changed: '+key)
                                if not np.array_equal(old['region_mean'][...,0,:,:],new['region_mean'][...,0,:,:]):raise ValueError('Untouched ViT attention changed')
                                for group in ['target_image','generation_markers']:
                                    ix=old['group_names'].tolist().index(group)
                                    if not np.array_equal(old['group_stats'][...,ix,:],new['group_stats'][...,ix,:]):raise ValueError('Untouched group changed: '+group)
                    image.save(d/f'turn_{turn}.png')
                    receipt={'session_id':sid,'turn':turn,'instruction':instructions[turn-1],'session_fingerprint':session['fingerprint'],
                        'input_hashes':source['input_hashes'],'output_hash':output_hash,'original_turn_json_sha256':sha(folder/f'turn_{turn}.json'),
                        'original_attention_sha256':sha(folder/f'turn_{turn}.attention.npz'),'source_model_settings':session['settings'],
                        'runtime_manifest_sha256':runtime_hash,'runtime_sources':runtime_sources,'recollection_settings':settings.identity(),
                        'backend':meta,'wall_seconds':time.time()-t,'mode':mode}
                    write(d/f'turn_{turn}.json',receipt); records.append({'session_id':sid,'turn':turn,'mode':mode,'attention_sha256':None if attn is None else sha(attn),'output_hash':output_hash})
                    write(args.output/'progress.json',{'status':'running','updated':time.time(),'session_id':sid,'turn':turn,'mode':mode,'completed_turns':len(records),'expected_sessions':len(selected)})
        write(args.output/'completion.json',{'status':'passed','hostname':socket.gethostname(),'format':'target-token-region-stats-v2','pilot':args.pilot,
            'runtime_manifest_sha256':runtime_hash,'runtime_sources':runtime_sources,'worker_script_sha256':sha(Path(__file__)),
            'sessions':len(selected),'turns':len(records),'all_outputs_exact':True,'all_real_identity_checks_passed':True,'records':records,'elapsed_seconds':time.time()-started})
    except BaseException as e:
        write(args.output/'failure.json',{'status':'failed','error':repr(e),'completed_turns':len(records),'updated':time.time()});raise
    finally:
        if backend is not None:backend.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--gates',type=Path)
    run(p.parse_args())

#!/usr/bin/env python3
"""GPU acceptance: both protocols, both resolutions, exact on/off pixel comparison."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from lance_mice.backend import OmniBackend
from lance_mice.dataset import load_samples
from lance_mice.protocol import digest,PROTOCOL_VERSION,CHAT_PROTOCOL_VERSION
from lance_mice.settings import Settings
from lance_mice.runner import run_session,write_json
from lance_mice.acceptance import validate_run

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--benchmark',choices=['edival','mice'],required=True);a=p.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
root=Path('/home/chs/dataset')/('EdiVal' if a.benchmark=='edival' else 'MICE-Bench')
sample=load_samples(root,'smoke',a.benchmark)[0]
s=Settings(resolution=512 if a.benchmark=='edival' else 768,cache_mode='prefix',attention_format='target-token-region-stats-v1')
b=OmniBackend(Path('/home/chs/model/Lance'),s,audit=True)
results={}
try:
 for proto in [PROTOCOL_VERSION,CHAT_PROTOCOL_VERSION]:
  b.settings=replace(s,history_protocol=proto)
  rows={}
  for record in [False,True]:
   out=a.output/proto/('record' if record else 'plain')
   spec={'settings':b.settings.identity(),'samples':[(sample.session_id,list(sample.instructions))]}
   if record:spec['attention']='target-token-region-stats-v1'
   rid=digest(spec);write_json(out/'run.json',{'fingerprint':rid,**spec})
   run_session(sample,out,b.settings,b,run_id=rid,save_attention=record)
   rows[record]=validate_run(out)
  if [x['output_hash'] for x in rows[False].values()]!=[x['output_hash'] for x in rows[True].values()]:raise ValueError('Observation changed generated pixels')
  results[proto]={'turns':len(rows[True]),'identical_pixels':True,'attention':[x['backend']['attention'] for x in rows[True].values()]}
finally:b.close()
write_json(a.output/'validation.json',{'status':'passed','benchmark':a.benchmark,'protocols':results})
print('GPU_ACCEPTANCE_PASSED',a.benchmark,flush=True)

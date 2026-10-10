#!/usr/bin/env python3
"""Own one fixed GPU collection process; clean temp and restore that card's burn."""
import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
ROOT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal');NAME='edival_semantic_20261009_v2'
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--host',choices=['a800_0','a800_1'],required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    expected={'a800_0':'aibox-r61097f954fe-7d74d99d65-2zzkn','a800_1':'aibox-r7df77faf487-f8c468557-b9c9s'}
    if socket.gethostname()!=expected[a.host]:raise ValueError('Wrong host')
    root=ROOT/'outputs/attention_recollection'/NAME;dispatch=root/'dispatch';dispatch.mkdir(parents=True,exist_ok=True)
    stage='pilot_r2' if a.pilot else f'full_gpu{a.gpu}';state=dispatch/f'{stage}.json'
    env=dict(os.environ);tmp=None;child=None;code=None;error=None
    try:
        used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>1000:raise ValueError('GPU not free before worker: '+str(used))
        tmp=tempfile.TemporaryDirectory(prefix=f'edival-semantic-{stage}-')
        runtime=ROOT/'runtime'/(NAME+'_r2')
        env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUDA_HOME='/usr/local/cuda-13.0',LD_LIBRARY_PATH='/usr/local/cuda-13.0/compat:/home/chs/conda/envs/lance/lib',PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=tmp.name,XDG_CACHE_HOME=tmp.name+'/cache',VLLM_CACHE_ROOT=tmp.name+'/vllm',TORCHINDUCTOR_CACHE_DIR=tmp.name+'/inductor',TRITON_CACHE_DIR=tmp.name+'/triton',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DO_NOT_TRACK='1',VLLM_NO_USAGE_STATS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_V2_MODEL_RUNNER='0',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',LANCE_AUDIT_ALL_LAYERS='1' if a.pilot else '0')
        command=['/home/chs/conda/envs/lance/bin/python','-B',str(ROOT/'scripts/recollect_edival_attention.py'),'--repo',str(ROOT),'--output',str(root/stage),'--shard',str(a.gpu)]
        if a.pilot:command.append('--pilot')
        else:command+=['--gates',str(root/'gates')]
        with (dispatch/f'{stage}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=tmp.name,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(state,{'status':'running','pid':child.pid,'gpu':a.gpu,'hostname':socket.gethostname(),'started':time.time(),'temporary_directory':tmp.name})
            code=child.wait()
    except BaseException as exc:error=repr(exc)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            # Worker-spawned processes share this owned process group. Stop only
            # this group if engine shutdown left descendants behind.
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        if tmp is not None:tmp.cleanup()
        restored=False
        for _ in range(12):
            used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            if used<=1000:
                burn='/media/damoxing/tangzecong/llamafactory_burn.sh' if a.host=='a800_0' else '/home/chs/tools/gpu-burn/llamafactory_burn.sh'
                # Refuse if the pane still has children; no duplicate burn loop.
                pane_pid=subprocess.check_output(['tmux','display-message','-p','-t',f'{a.gpu}:0.0','#{pane_pid}'],text=True).strip()
                children=subprocess.run(['pgrep','-P',pane_pid],capture_output=True,text=True).stdout.strip()
                if children:error=(error or '')+' burn pane occupied';break
                cmd=f'BURN_STEPS=2000 BURN_BATCH=8 BURN_CUTOFF_LEN=4096 BURN_IMAGE_PIXELS=1572864 BURN_LORA_RANK=128 bash {burn} {a.gpu}'
                subprocess.run(['tmux','send-keys','-t',f'{a.gpu}:0.0',cmd,'C-m'],check=True);restored=True;break
            time.sleep(5)
        write(state,{'status':'passed' if code==0 and error is None else 'failed','exit_code':code,'error':error,'gpu':a.gpu,'hostname':socket.gethostname(),'finished':time.time(),'temporary_directory_removed':True,'burn_restart_sent':restored})
    if code!=0 or error:raise SystemExit(1)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Execute installed attention cache-assembly methods with CPU identity probes.

Only AST-extracted methods execute. QKV, norm, RoPE, attention and projection
are deterministic stubs: this tests token identity/order, not model behavior.
No weights, CUDA, inference outputs, runtime patches or source edits.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract(path, cls, methods):
    tree=ast.parse(path.read_text())
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return [n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in methods]


class Probe:
    num_heads=num_kv_heads=1
    head_dim=q_size=kv_size=hidden_size=2
    layer_idx=0

    def __init__(self):
        self.captured_keys=[]
        self.qkv_proj=lambda x,*args:(torch.cat([x,x,x],-1),None)
        self.q_norm=self.k_norm=lambda x,*args:x
        self.rotary_op=lambda x,*args:x
        self.o_proj=lambda x,*args:(x,None)
        self.attn_noncausal_local=self.attn_causal=self.attend

    def _is_sp_active(self):return False

    def attend(self,q,k,v,*args):
        self.captured_keys.append(k[0,:,0,0].float().tolist())
        return torch.zeros_like(q)


def main():
    base=Path(sys.prefix)/'lib/python3.12/site-packages/vllm_omni/diffusion/models'
    source=base/'bagel/bagel_transformer.py'
    lance=base/'lance/lance_transformer.py'
    methods=extract(source,'PackedAttentionMoT',{'_forward_gen','_forward_und'})
    env=dict(torch=torch,NaiveCache=type('CacheHelpers',(),{'split_with_zeros':staticmethod(lambda x,lens:x.split(lens))}),
             left_pad_stack=lambda parts:(torch.stack(parts),None),DiffusionAttentionMetadata=lambda **kwargs:kwargs)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*methods],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),env)
    for m in methods:setattr(Probe,m.name,env[m.name])
    p=Probe();cache=SimpleNamespace(key_cache=[None],value_cache=[None],key_values_lens=[0])
    def inp(ids):return torch.tensor([[i,i] for i in ids],dtype=torch.float32)
    def emb(n):return (torch.ones(n,2),torch.zeros(n,2))
    vit=[11,31,32,33,34,21]
    p._forward_und(inp(vit),emb(6),cache,False,True)
    assert cache.key_cache[0][:,0,0].tolist()==vit
    cache.key_values_lens=[6]
    vae=[12,41,42,43,44,22]
    p._forward_gen(inp(vae),torch.tensor([6]),emb(6),cache,
                   torch.tensor([1,2,3,4]),torch.tensor([0,5]),True)
    actual=cache.key_cache[0][:,0,0].float().tolist()
    expected=vit+[12,22,41,42,43,44]
    assert actual==expected
    # What the saved spatial range [begin+1, end-1) actually selects.
    misclassified=actual[7:11]
    assert misclassified==[22,41,42,43]
    und=extract(source,'PackedAttentionMoT',{'_forward_und'})[0]
    gen=extract(source,'PackedAttentionMoT',{'_forward_gen'})[0]
    proj=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
    runtime=proj/'runtime/sixrun_20261007_tokenregion_v3/src/lance_mice'
    evidence=dict(status='reproduced',hostname=socket.gethostname(),cpu_only=True,
        test='Installed method bodies, deterministic identity stubs, one ViT then one VAE prefill',
        limitation='Confirms cache-order logic only; no actual attention magnitude or model inference measured',
        vit_input=vit,vae_input=vae,actual_cache=actual,
        observer_vae_spatial_ids=misclassified,true_vae_spatial_ids=[41,42,43,44],
        false_spatial_id=22,excluded_spatial_id=44,
        source_files={str(p):sha(p) for p in [source,lance,runtime/'omni_pipeline.py',runtime/'attention_regions.py',runtime/'attention.py']},
        method_ranges={'_forward_gen':[gen.lineno,gen.end_lineno],'_forward_und':[und.lineno,und.end_lineno]},
        script_sha256=sha(Path(__file__)),torch=torch.__version__)
    out=proj/'outputs/attention_analysis/edival_region_bias_20261009_v1/cache_order_audit.json'
    out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()

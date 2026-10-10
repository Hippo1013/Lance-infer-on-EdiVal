"""Read-only, real-tensor cache identity checks for the pinned Lance implementation.

Hooks capture post-RoPE Q/K before the native gather, then compare exact BF16
values with native cached keys and positive denoising attention inputs. No RNG,
model tensors or numerical operations used by generation are changed.
"""
from __future__ import annotations
import inspect
import types
import hashlib
import json

class CacheIdentityAudit:
    def __init__(self, bagel, all_layers=False):
        import torch
        self.torch=torch
        layers=sorted((m.layer_idx,m) for m in bagel.modules() if hasattr(m,'_forward_gen') and hasattr(m,'attn_noncausal_local'))
        self.layers=layers if all_layers else [layers[0],layers[-1]]
        self.states={}; self.orders={}; self.counts={}; self.handles=[]
        self.reset()
        for idx,module in self.layers:
            self.states[idx]=None
            self.handles.append(module.rotary_op.register_forward_hook(self._rotary(idx)))
            self.handles.append(module.attn_noncausal_local.register_forward_pre_hook(self._attention(idx),with_kwargs=True))
            for mode in ('gen','und'):
                name='_forward_'+mode; original=getattr(module,name); signature=inspect.signature(original)
                def wrapped(this,*args,_idx=idx,_mode=mode,_original=original,_sig=signature,**kwargs):
                    bound=_sig.bind(*args,**kwargs); bound.apply_defaults(); p=bound.arguments
                    cache=p.get('past_key_values'); old=None if cache is None else cache.key_cache[_idx]
                    s={'mode':_mode,'p':p,'old':old,'rope':[]}; self.states[_idx]=s
                    try:
                        result=_original(*args,**kwargs)
                        if p['update_past_key_values']:
                            k=s['rope'][1]
                            if _mode=='gen':
                                order=self.torch.cat([p['packed_text_indexes'],p['packed_vae_token_indexes']])
                                k=k[order]
                                # Preserve the actual native gather indices for spatial trace construction.
                                self.orders[_idx]=order.cpu().tolist()
                            actual=result[1].key_cache[_idx]
                            start=0 if old is None else len(old)
                            self._equal(actual[start:],k,'cache_append',_idx)
                            if old is not None:self._equal(actual[:start],old,'cache_prefix',_idx)
                        return result
                    finally:self.states[_idx]=None
                setattr(module,name,types.MethodType(wrapped,module))
    def reset(self):
        self.counts={}; self.orders={}
    def _equal(self,a,b,kind,idx):
        if a.shape!=b.shape or not self.torch.equal(a,b):
            raise ValueError(f'Real token identity failure: {kind}, layer {idx}, {tuple(a.shape)}, {tuple(b.shape)}')
        key=f'{kind}:layer{idx}'; self.counts[key]=self.counts.get(key,0)+1
    def _rotary(self,idx):
        def hook(module,args,out):
            s=self.states[idx]
            if s is None:raise ValueError('Untracked rotary operation')
            if len(s['rope'])>=2:raise ValueError('Unexpected rotary call sequence')
            s['rope'].append(out.squeeze(0).detach().to(self.torch.bfloat16))
        return hook
    def _attention(self,idx):
        def hook(module,args,kwargs):
            s=self.states[idx]
            if s is None or s['mode']!='gen':return
            p=s['p']; branches=len(p['query_lens']); q,k=s['rope']; ti=p['packed_text_indexes']; vi=p['packed_vae_token_indexes']
            nt=len(ti)//branches; nv=len(vi)//branches
            text=ti[:nt]; vae=vi[:nv]
            positive_q=self.torch.cat([q[text],q[vae]])
            positive_k=self.torch.cat([k[text],k[vae]])
            if s['old'] is not None:
                lens=getattr(p['past_key_values'],'key_values_lens',None)
                n=lens[0] if lens is not None else len(s['old'])//branches
                positive_k=self.torch.cat([s['old'][:n],positive_k])
            actual_q,actual_k=args[:2]
            self._equal(actual_q[0],positive_q,'positive_query',idx)
            self._equal(actual_k[0],positive_k,'positive_key',idx)
            if not p['update_past_key_values']:
                if nt!=2 or nv!=1024 or text.cpu().tolist()!=[0,1025] or vae.cpu().tolist()!=list(range(1,1025)):
                    raise ValueError('Target-query token identity differs from pinned 512px layout')
                self._equal(actual_q[0,2:],q[vae],'target_query',idx)
        return hook
    def image_cache_order(self):
        values=list(self.orders.values())
        if len(values)!=len(self.layers) or any(x!=values[0] for x in values):
            raise ValueError('Missing or inconsistent real VAE cache identity evidence')
        return values[0]
    def evidence(self):
        return {'status':'passed','method':'exact BF16 equality between post-RoPE native input identities, actual cache append, positive Q/K and target queries',
                'layers':[i for i,_ in self.layers],'checks':dict(self.counts),
                'native_vae_gather_sha256':hashlib.sha256(json.dumps(self.orders,sort_keys=True).encode()).hexdigest()}

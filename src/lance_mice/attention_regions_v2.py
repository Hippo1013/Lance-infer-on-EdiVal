"""Positive denoising attention: text tokens, 8x8 ViT/VAE regions and query statistics.

Observation uses detached FP16 Q/K and fused SDPA indicator values. Full-key
softmax, FP32 reductions, no modification of generation tensors or RNG.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .attention import AttentionObserver, group_layout, reference_mass

ATTENTION_VERSION = 'target-token-region-stats-v2'
STAT_NAMES = ['mean', 'std', 'p10', 'p50', 'p90']


def region_layout(trace, context_tokens, marker_tokens, target_tokens):
    names, labels, _ = group_layout(trace, context_tokens, marker_tokens, target_tokens)
    fine = np.full(len(labels), -1, dtype=np.int64)
    images = sorted({x['image_index'] for x in trace if x['kind'] in {'vit', 'vae'}})
    if images != list(range(len(images))):
        raise ValueError('Noncontiguous image history')
    regions = len(images) * 2 * 64
    text_keys, token_ids, token_turns, token_offsets = [], [], [], []
    geometry = []
    for x in trace:
        if x['kind'] in {'vit', 'vae'}:
            a, b = x['tokens']; lo, hi = x['spatial_tokens']; h, w = x['grid_hw']
            expected = (a+1,b-1) if x['kind']=='vit' else (a+2,b)
            if (lo, hi) != expected or hi-lo != h*w or not x.get('cache_identity_verified'):
                raise ValueError('Spatial grid/marker mapping mismatch')
            markers = [a,b-1] if x['kind']=='vit' else [a,a+1]
            if x.get('marker_key_positions') != markers:raise ValueError('Image marker identity mismatch')
            labels[markers] = 0
            # Assign patch centers to half-open bins of the normalized image.
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            cell = ((2*yy+1)*8//(2*h))*8 + ((2*xx+1)*8//(2*w))
            base = (x['image_index']*2 + int(x['kind']=='vae'))*64
            fine[lo:hi] = base + cell.ravel()
            geometry.append(x)
        elif x.get('role') in {'history','current'}:
            a, b = x['tokens']
            if len(x['token_ids']) != b-a or len(x['offsets']) != b-a:
                raise ValueError('Instruction token index mismatch')
            text_keys.extend(range(a,b)); token_ids.extend(x['token_ids'])
            token_turns.extend([x['turn']]*(b-a)); token_offsets.extend(x['offsets'])
    for i, pos in enumerate(text_keys): fine[pos] = regions+i
    marker_positions=[]
    for x in geometry:
        marker_positions.extend(x['marker_key_positions'])
    marker_labels=np.full(len(labels),-1,dtype=np.int64)
    for i,pos in enumerate(marker_positions):marker_labels[pos]=i
    region_counts = np.bincount(fine[fine>=0], minlength=regions+len(text_keys))[:regions]
    counts = np.bincount(labels, minlength=len(names))
    return dict(names=names, labels=labels, counts=counts, fine=fine, regions=regions,
        image_count=len(images), region_counts=region_counts.reshape(len(images),2,8,8),
        text_keys=np.array(text_keys,dtype=np.int64), token_ids=np.array(token_ids,dtype=np.int64),
        token_turns=np.array(token_turns,dtype=np.int64), token_offsets=np.array(token_offsets,dtype=np.int64),
        geometry=geometry,marker_positions=np.array(marker_positions,dtype=np.int64),marker_labels=marker_labels)


def indicator_queries(q, k, labels, start, count):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    width=q.shape[-1]
    basis=torch.zeros(k.shape[-2],width,dtype=q.dtype,device=q.device)
    selected=(labels>=start)&(labels<start+count)
    idx=selected.nonzero().flatten()
    basis[idx,labels[idx]-start]=1
    v=basis[None,None].expand(1,k.shape[1],-1,-1)
    with torch.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=False,
            scale=q.shape[-1]**-0.5,enable_gqa=q.shape[1]!=k.shape[1])[0,:,:,:count].float()


class RegionAttentionObserver(AttentionObserver):
    def __init__(self, bagel, trace, context_tokens, marker_tokens, target_tokens, steps):
        super().__init__(bagel,trace,context_tokens,marker_tokens,target_tokens,steps)
        self.layout=region_layout(trace,context_tokens,marker_tokens,target_tokens)
        self.names=self.layout['names'];self.labels=self.layout['labels'];self.counts=self.layout['counts']
        self.mapping=None

    def observe(self,q,k,index):
        import torch
        q=q.to(torch.float16);k=k.to(torch.float16)
        if self.mapping is None:
            self.mapping=(torch.as_tensor(self.labels,device=q.device),torch.as_tensor(self.layout['fine'],device=q.device),torch.as_tensor(self.layout['marker_labels'],device=q.device))
        labels,fine,marker_labels=self.mapping
        per_query=indicator_queries(q,k,labels,0,len(self.names))
        quant=torch.quantile(per_query,torch.tensor([.1,.5,.9],device=q.device),dim=1)
        stats=torch.stack([per_query.mean(1),per_query.std(1,correction=0),quant[0],quant[1],quant[2]],-1)
        n=self.layout['regions']+len(self.layout['text_keys'])
        parts=[]
        for start in range(0,n,q.shape[-1]):
            val=indicator_queries(q,k,fine,start,min(q.shape[-1],n-start))
            parts.append(val.mean(1))
        mean=torch.cat(parts,-1)
        marker_mean=indicator_queries(q,k,marker_labels,0,len(self.layout['marker_positions'])).mean(1)
        if index in (self.layers[0][0],self.layers[-1][0]) and self.step in (0,self.steps-1):
            sample=q[:,:,[0,self.target_tokens//2,self.target_tokens-1],:]
            measured=indicator_queries(sample,k,labels,0,len(self.names)).mean(1)
            expected=reference_mass(sample,k,labels,len(self.names))
            self.errors.append((measured-expected).abs().max().cpu())
            # Independently evaluate all fine channels from sampled actual Q/K.
            kk=k.float().repeat_interleave(sample.shape[1]//k.shape[1],dim=1)
            with torch.autocast('cuda',enabled=False):
                probs=(sample.float()@kk.transpose(-1,-2)*sample.shape[-1]**-.5).softmax(-1)[0]
            chosen=fine>=0
            ref=torch.zeros(probs.shape[0],probs.shape[1],n,device=q.device)
            ref.scatter_add_(2,fine[chosen][None,None].expand(probs.shape[0],probs.shape[1],-1),probs[...,chosen])
            observed=torch.cat([indicator_queries(sample,k,fine,a,min(q.shape[-1],n-a)).mean(1) for a in range(0,n,q.shape[-1])],-1)
            self.errors.append((observed-ref.mean(1)).abs().max().cpu())
        self.rows[(self.step,index)]=(stats.cpu().numpy(),mean.cpu().numpy(),marker_mean.cpu().numpy())

    def __enter__(self):
        import torch
        self.original_forward=self.bagel.forward
        def forward(*args,**kwargs):
            if args:raise ValueError('Keyword denoising interface required')
            self.step+=1;self.timesteps.append(kwargs['timestep'][0].detach().clone())
            return self.original_forward(**kwargs)
        self.bagel.forward=forward
        for index,layer in self.layers:
            def hook(module,args,kwargs,index=index):
                q,k=args[:2]
                if q.ndim!=4 or q.shape[1]!=self.marker_tokens+self.target_tokens or k.shape[1]!=len(self.labels):
                    raise ValueError('Unexpected positive attention layout')
                metadata=args[3] if len(args)>3 else kwargs.get('attn_metadata')
                mask=getattr(metadata,'attn_mask',None)
                if mask is not None:
                    mask=mask[0]
                    if not bool(mask.all() if mask.dtype==torch.bool else (mask==0).all()):
                        raise ValueError('Masked positive keys')
                if (self.step,index) in self.rows:raise ValueError('Duplicate attention row')
                self.observe(q[:1,self.marker_tokens:].permute(0,2,1,3).detach(),k[:1].permute(0,2,1,3).detach(),index)
            self.handles.append(layer.register_forward_pre_hook(hook,with_kwargs=True))
        return self

    def save(self,path):
        import torch
        if self.step+1!=self.steps or len(self.rows)!=self.steps*len(self.layers):raise ValueError('Incomplete observations')
        rows=[self.rows[(s,l)] for s in range(self.steps) for l,_ in self.layers]
        stats=np.stack([x[0] for x in rows]).reshape(self.steps,len(self.layers),-1,len(self.names),5)
        means=np.stack([x[1] for x in rows]).reshape(self.steps,len(self.layers),stats.shape[2],-1)
        mass=stats[...,0]
        err=max(x.item() for x in self.errors);norm=float(np.abs(mass.sum(-1)-1).max())
        if not np.isfinite(stats).all() or not np.isfinite(means).all() or stats.min()<0 or means.min()<0 or err>.0005 or norm>.001:
            raise ValueError(f'Attention probability check failed: {err=} {norm=}')
        regions=means[...,:self.layout['regions']].reshape(*means.shape[:3],self.layout['image_count'],2,8,8)
        # Regional cells sum to the corresponding spatial-only image group.
        for im in range(self.layout['image_count']):
            for typ,kind in enumerate(['vit','vae']):
                expected=mass[...,self.names.index(f'I{im}_{kind}')]
                if np.abs(regions[...,im,typ,:,:].sum((-1,-2))-expected).max()>.001:
                    raise ValueError('Regional mass conservation failure')
        path=Path(path)
        with path.open('xb') as f:
            np.savez(f,group_stats=stats,text_mean=means[...,self.layout['regions']:self.layout['regions']+len(self.layout['text_keys'])],region_mean=regions,
                image_marker_mean=np.stack([x[2] for x in rows]).reshape(*means.shape[:3],self.layout['image_count'],2,2),
                image_marker_key_positions=self.layout['marker_positions'],
                group_names=np.array(self.names),stat_names=np.array(STAT_NAMES),group_token_counts=self.counts,
                token_group_ids=self.labels,region_token_counts=self.layout['region_counts'],
                fine_key_channel=self.layout['fine'],text_key_positions=self.layout['text_keys'],
                text_token_ids=self.layout['token_ids'],text_token_turns=self.layout['token_turns'],text_token_offsets=self.layout['token_offsets'],
                image_geometry_json=np.array(json.dumps(self.layout['geometry'])),
                timesteps=torch.stack(self.timesteps).float().cpu().numpy(),layer_ids=np.array([l for l,_ in self.layers]),
                target_query_count=self.target_tokens,protocol=np.array(ATTENTION_VERSION))
        return dict(version=ATTENTION_VERSION,file=path.name.removesuffix('.part'),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            branch='positive',shape=list(mass.shape),axes=['step','layer','head','group'],groups=self.names,
            group_token_counts=self.counts.tolist(),target_query_count=self.target_tokens,
            text_shape=list(means[...,self.layout['regions']:self.layout['regions']+len(self.layout['text_keys'])].shape),region_shape=list(regions.shape),
            max_reference_absolute_error=err,max_probability_sum_error=norm,
            mean_mass=dict(zip(self.names,mass.mean((0,1,2)).tolist())),
            method='Detached post-norm/RoPE FP16 Q/K; full-key fused SDPA indicator sums; FP32 mean/population std/linear quantiles; patch-center 8x8 regions; positive branch; uncompressed FP32 arrays')


def validate_attention(path,metadata,trace,context_tokens,expected_steps):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('Attention hash mismatch')
    layout=region_layout(trace,context_tokens,metadata['group_token_counts'][-2],metadata['target_query_count'])
    with np.load(path,allow_pickle=False) as d:
        stats=d['group_stats'];mass=stats[...,0]
        if str(d['protocol'])!=ATTENTION_VERSION or list(mass.shape)!=metadata['shape'] or stats.dtype!=np.float32:
            raise ValueError('Format mismatch')
        for key,actual in [('group_token_counts',layout['counts']),('token_group_ids',layout['labels']),('region_token_counts',layout['region_counts']),('fine_key_channel',layout['fine']),('text_key_positions',layout['text_keys']),('text_token_ids',layout['token_ids']),('text_token_turns',layout['token_turns']),('text_token_offsets',layout['token_offsets'])]:
            if not np.array_equal(d[key],actual):raise ValueError('Index mismatch: '+key)
        if d['group_names'].tolist()!=layout['names'] or d['stat_names'].tolist()!=STAT_NAMES:raise ValueError('Label mismatch')
        for key in ['group_stats','region_mean','text_mean']:
            a=d[key]
            if a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<0:raise ValueError('Invalid numeric array')
        if mass.shape[0]!=expected_steps or np.abs(mass.sum(-1)-1).max()>.001:raise ValueError('Normalization failure')
        if not np.all(np.diff(d['timesteps'])<0):raise ValueError('Step order mismatch')
        if not np.all(stats[...,2]<=stats[...,3]) or not np.all(stats[...,3]<=stats[...,4]):raise ValueError('Quantile ordering')
        for im in range(layout['image_count']):
            for typ,kind in enumerate(['vit','vae']):
                if np.abs(d['region_mean'][...,im,typ,:,:].sum((-1,-2))-mass[...,layout['names'].index(f'I{im}_{kind}')]).max()>.001:raise ValueError('Region conservation')
        for x in trace:
            if x.get('role') in {'history','current'}:
                select=layout['token_turns']==x['turn']
                if np.abs(d['text_mean'][...,select].sum(-1)-mass[...,layout['names'].index(f'T{x["turn"]}')]).max()>.001:raise ValueError('Text conservation')
    return {'shape':metadata['shape'],'groups':layout['names'],'max_probability_sum_error':metadata['max_probability_sum_error']}

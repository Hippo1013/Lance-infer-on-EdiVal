"""Lance history pipeline with independently checked cache-space observation labels."""
import os
from .omni_pipeline import LanceHistoryPipeline
from .cache_semantics import CacheIdentityAudit

class SemanticLancePipeline(LanceHistoryPipeline):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.identity_audit=CacheIdentityAudit(self.bagel,all_layers=os.environ.get('LANCE_AUDIT_ALL_LAYERS')=='1')
    def _prefill_image(self,ctx,image,image_index,session_id,settings,target_size,trace):
        self.identity_audit.orders={}
        super()._prefill_image(ctx,image,image_index,session_id,settings,target_size,trace)
        vit,vae=trace[-2:]
        for x in (vit,vae):x['input_spatial_tokens']=list(x['spatial_tokens'])
        a,b=vit['tokens']; vit['marker_key_positions']=[a,b-1]; vit['cache_order']='input-order'
        a,b=vae['tokens']; order=self.identity_audit.image_cache_order()
        if order!=[0,b-a-1]+list(range(1,b-a-1)):
            raise ValueError('Native VAE gather differs from validated marker-first schema')
        vae['marker_key_positions']=[a,a+1]; vae['spatial_tokens']=[a+2,b]; vae['cache_order']='text-markers-then-spatial'
        for x in (vit,vae):x['cache_identity_verified']=True
    def _forward_history(self,req,history):
        self.identity_audit.reset()
        result=super()._forward_history(req,history)
        evidence=self.identity_audit.evidence()
        # Every step must have positive/target checks on every audited layer.
        steps=history['settings']['steps']
        for layer in evidence['layers']:
            if evidence['checks'].get(f'target_query:layer{layer}')!=steps:
                raise ValueError('Incomplete denoising identity audit')
        result.output['metadata']['lance_history']['cache_identity_audit']=evidence
        return result

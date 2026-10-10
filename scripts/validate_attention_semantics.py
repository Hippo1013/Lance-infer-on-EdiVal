#!/usr/bin/env python3
"""Independent CPU mapping regression: marker-first VAE versus sequential ViT."""
from lance_mice.attention_regions_v2 import region_layout
import numpy as np

def main():
    trace=[dict(kind='vit',image_index=0,tokens=[0,6],spatial_tokens=[1,5],grid_hw=[2,2],marker_key_positions=[0,5],cache_identity_verified=True),
           dict(kind='vae',image_index=0,tokens=[6,12],spatial_tokens=[8,12],grid_hw=[2,2],marker_key_positions=[6,7],cache_identity_verified=True),
           dict(kind='text',role='current',turn=1,tokens=[13,15],token_ids=[100,200],offsets=[[0,1],[1,2]])]
    l=region_layout(trace,15,2,4)
    actual=np.array([11,31,32,33,34,21,12,22,41,42,43,44,900,100,200,91,92,51,52,53,54])
    for kind,ids in [('vit',[31,32,33,34]),('vae',[41,42,43,44])]:
        group=l['names'].index('I0_'+kind)
        assert actual[l['labels']==group].tolist()==ids
    assert l['counts'][0]==5
    assert np.array_equal(l['region_counts'].sum((-1,-2)),[[4,4]])
    assert actual[l['marker_positions']].tolist()==[11,21,12,22]
    assert actual[l['text_keys']].tolist()==[100,200]
    trace[1]['spatial_tokens']=[7,11]
    try:region_layout(trace,15,2,4)
    except ValueError:pass
    else:raise AssertionError('Old erroneous VAE map was accepted')
    print('PASS: true synthetic cache identities, marker exclusion, region cardinality, text separation and old-mapping rejection')
if __name__=='__main__':main()

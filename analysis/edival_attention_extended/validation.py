"""Independent mathematical fixtures for concentration, participation and pairing.

Use a system temporary directory for test output and Python bytecode caches.
These fixtures do not establish real-data coverage or acceptance.
"""
import math
import numpy as np
from core import concentration, distribution, entropy, mean_valid, ratio, tv, js, top_overlap, compact, split_clusters


def test_formulas():
    # Independent scalar calculations, including uneven/zero support and one token.
    x=np.array([[1.,1.,1.,1.],[8.,1.,1.,0.],[0.,0.,0.,0.]])
    mass,p,metrics=concentration(x)
    assert metrics['entropy'][0]==1
    scalar_p=[.8,.1,.1]
    scalar_h=-sum(v*math.log(v) for v in scalar_p)/math.log(4)
    assert abs(metrics['entropy'][1]-scalar_h)<1e-14
    assert metrics['top10'][1]==.8 and metrics['n90'][1]==.5
    assert np.isnan(metrics['entropy'][2]) and mass[2]==0
    assert np.isnan(concentration(np.array([[2.]]))[2]['entropy'][0])
    assert np.allclose(tv(p[:1],p[1:2]),[.55])
    assert np.allclose(js(p[:1],p[:1]),[0])
    assert np.allclose(top_overlap(p[1:2],p[1:2],2),[1])
    # Equal conditional means and amount weighting answer distinct questions.
    m=np.array([1.,9.]);s=np.array([.9,.1])
    assert abs(float(mean_valid(s))-.5)<1e-14
    assert abs(float(ratio((m*s).sum(),m.sum()))-.18)<1e-14
    mu=np.array([.2,.2]);sd=np.array([0.,math.sqrt(.12)])
    r=ratio(mu*mu,mu*mu+sd*sd)
    assert np.allclose(r,[1.,.25])
    assert np.isnan(ratio(0.,0.))
    items=[{'session_id':str(k),'input_hashes':[str(k//2)]} for k in range(8)]
    split=split_clusters(items)
    assert split['0']['split']==split['1']['split']
    assert len({v['cluster'] for v in split.values() if v['split']=='exploration'})==2
    # Nonlinear per-head entropy differs from entropy of mean distribution.
    a=np.array([[1.,0.],[0.,1.]])
    assert abs(float(entropy(a.mean(0)))-math.log(2))<1e-14
    assert float(entropy(a).mean())==0
    return {'scalar_fixtures':'passed','fixtures':11}


if __name__=='__main__':print(test_formulas())

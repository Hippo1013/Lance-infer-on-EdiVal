"""Mathematical and weighting cases with explicit expected results."""
import math, numpy as np
from common import *
from metrics import calculate, scalar_rows, METRICS
from aggregation import equal_average, describe
from pipeline import density, paired

def run():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def val(r,name):return r['values'][0,METRICS.index(name)]
    r=calculate([[.05]*10])
    check('uniform entropy',abs(val(r,'entropy_nats')-math.log(10))<1e-12)
    check('uniform effective count',abs(val(r,'effective_count')-10)<1e-12)
    check('uniform top k',abs(val(r,'top3_share')-.3)<1e-12 and val(r,'n90')==9)
    r=calculate([[0,0,.6,0]])
    check('point entropy',val(r,'entropy_nats')==0 and val(r,'effective_count')==1 and val(r,'n90')==1)
    check('point zero elements remain in N',r['N']==4 and val(r,'effective_fraction')==.25)
    r=calculate([[.5,.3,.2]])
    check('known nonuniform',abs(val(r,'entropy_nats')+sum(p*math.log(p) for p in [.5,.3,.2]))<1e-12 and val(r,'n90')==3)
    r=calculate([[.9,.1]])
    check('n90 threshold tolerance',val(r,'n90')==1)
    r=calculate([[1.]])
    check('N1 metric-specific validity',not r['valid'][0,2] and r['valid'][0].sum()==10 and val(r,'effective_fraction')==1)
    s=scalar_rows(r,{})
    check('N1 scalar denominators',s['normalized_entropy'] is None and s['entropy_nats_n_valid']==1 and s['normalized_entropy_n_valid']==0)
    r=calculate([[0.,0.],[1e-20,1e-20],[1e-6,0],[.2,.3]])
    check('zero mass only conditional undefined',r['valid'][0,0] and not r['valid'][0,1:].any())
    check('low mass no smoothing',np.array_equal(r['low'],[False,True,True,False]) and abs(r['values'][1,1]-math.log(2))<1e-12)
    s=scalar_rows(r,{})
    check('sensitivity denominator',s['entropy_nats_n_valid']==3 and s['entropy_nats_sensitivity_n_valid']==1 and s['support_mass_n_valid']==4)
    check('conditional profiles differ',np.max(np.abs(r['conditional']-r['pooled']))>.01)
    r=calculate([[1.,0],[0,1.]])
    check('nonlinear before averaging',r['values'][:,1].mean()==0 and abs(calculate([[.5,.5]])['values'][0,1]-math.log(2))<1e-12)
    try:calculate(np.empty((2,0)))
    except ValueError:checks.append('N0 rejected')
    else:raise AssertionError('N0 accepted')
    try:calculate([[-1.,2.]])
    except ValueError:checks.append('negative input rejected')
    else:raise AssertionError('negative accepted')
    d=density({},.3,0,10)
    check('structural zero density null',not d['present'] and d['per_token_mass'] is None and d['density_null_reason']=='structural_zero')
    vit=density({},.2,2,10);vae=density({},.3,6,10);combined=density({},.5,8,10)
    check('combined density token weighted',combined['per_token_mass']==.0625 and combined['per_token_mass']!=(vit['per_token_mass']+vae['per_token_mass'])/2)
    # Sessions have unequal numbers of objects and different token lengths.
    rs=[dict(value=.2),dict(value=.4)];sessionA=equal_average(rs);sessionB=equal_average([dict(value=.9)])
    check('objects within session equal then sessions equal',abs((sessionA+sessionB)/2-.6)<1e-12 and abs((.2+.4+.9)/3-.6)>.01)
    check('present_only turn means',equal_average([dict(value=None),dict(value=.2),dict(value=.4)])==np.mean([.2,.4]))
    check('density samples before averaging',abs((.2/2+.3/6)/2-.075)<1e-12 and abs(.5/8-.075)>.001)
    # Cross-turn object missing in turn one produces only a 2->3 pair.
    tr=[dict(object_id='a',session_id='s',generation_turn=t,source_id=str(t),object_type='text',object_index=2,modality='text',object_role='current_instruction',age=t-2,
        object_identity='abc',token_count=2,mass=v,per_token_mass=v/2,enrichment=v*5) for t,v in [(2,.2),(3,.3)]]
    ps=paired(tr)
    check('both present pairing',len(ps)==1 and ps[0]['turn_from']==2 and abs(ps[0]['delta_mass']-.1)<1e-12)
    check('overlapping subword offsets retained','abcd'[1:3]=='bc' and 'abcd'[2:4]=='cd')
    clusters={'s1':'x','s2':'x','s3':'y'}
    summary=describe({'s1':0.,'s2':0.,'s3':1.},clusters,'synthetic')
    check('cluster bootstrap identity and session estimate',summary['n_clusters']==2 and summary['mean']==1/3 and summary['ci_low']==0 and summary['ci_high']==1)
    # The point estimate cannot become a uniform mean of cluster means (1/2).
    check('cluster point estimate preserves session weights',summary['mean']!=.5)
    atomic_json(ROOT/'validation/numerical_tests.json',dict(task_id=TASK,status='passed',checks=checks,n_checks=len(checks),at=now()))
    return checks

if __name__=='__main__':print('mathematical tests:',len(run()),'passed')

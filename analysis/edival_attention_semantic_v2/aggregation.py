"""Session equal weighting and original-image cluster percentile bootstrap."""
from collections import defaultdict
import hashlib
import numpy as np

SEED=20261008
REPS=2000
QUANTILES=[.1,.25,.5,.75,.9]

def describe(values, clusters, key, n_total=None, n_structural_zero=0):
    finite=[(s,float(v)) for s,v in sorted(values.items()) if v is not None]
    if any(not np.isfinite(v) for s,v in finite):raise ValueError('Nonfinite summary')
    n=len(finite); total=len(values) if n_total is None else n_total
    result=dict(n_total=total,n_sessions=total,n_valid=n,n_structural_zero=n_structural_zero,
        n_clusters=len(set(clusters[s] for s,v in finite)),bootstrap_reps=REPS,bootstrap_seed=SEED,
        bootstrap_unit='original_image_cluster',bootstrap_method='resample clusters uniformly with replacement; include all their sessions; ratio of sampled session sums/counts; percentile',
        mean=None,std=None,p10=None,p25=None,p50=None,p75=None,p90=None,min=None,max=None,ci_low=None,ci_high=None)
    if not n:return result
    x=np.array([v for s,v in finite],dtype=np.float64)
    result.update(mean=float(x.mean()),std=float(x.std(ddof=1)) if n>1 else None,min=float(x.min()),max=float(x.max()))
    result.update(dict(zip(['p10','p25','p50','p75','p90'],map(float,np.quantile(x,QUANTILES)))))
    grouped=defaultdict(list)
    for s,v in finite:grouped[clusters[s]].append(v)
    ordered=sorted(grouped); sums=np.array([sum(grouped[g]) for g in ordered]);counts=np.array([len(grouped[g]) for g in ordered])
    # The keyed stream is independent of process completion order.
    salt=int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8],'little')
    rng=np.random.default_rng(np.random.SeedSequence([SEED,salt]))
    boot=[]
    for start in range(0,REPS,100):
        picks=rng.integers(0,len(ordered),size=(min(100,REPS-start),len(ordered)))
        boot.extend((sums[picks].sum(1)/counts[picks].sum(1)).tolist())
    result['ci_low'],result['ci_high']=map(float,np.quantile(boot,[.025,.975]))
    return result

def equal_average(rows, value='value'):
    present=[r[value] for r in rows if r[value] is not None]
    return float(np.mean(present,dtype=np.float64)) if present else None

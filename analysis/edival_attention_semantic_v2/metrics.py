"""FP64 support metrics; nonlinear functions are evaluated at each position."""
import math
import numpy as np

METRICS = ['support_mass','entropy_nats','normalized_entropy','effective_count','effective_fraction','top1_share','top3_share','top5_share','top10pct_share','n90','n90_fraction']
LOW_MASS=1e-6
N90_TOL=1e-12

def calculate(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]==0 or not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid support')
    B,N=x.shape; M=x.sum(1); positive=M>0; low=positive & (M<=LOW_MASS)
    p=np.zeros_like(x); np.divide(x,M[:,None],out=p,where=positive[:,None])
    logs=np.zeros_like(p); np.log(p,out=logs,where=p>0)
    H=-(p*logs).sum(1)
    ordered=np.sort(p,axis=1)[:,::-1]
    csum=np.cumsum(ordered,axis=1)
    n90=np.argmax(csum >= .9-N90_TOL,axis=1)+1
    if positive.any() and not np.all(csum[positive,-1]>=.9-N90_TOL):raise ValueError('n90 normalization failure')
    ks=[1,min(3,N),min(5,N),math.ceil(.1*N)]
    values=np.column_stack([M,H,H/math.log(N) if N>1 else np.zeros(B),np.exp(H),np.exp(H)/N,*[csum[:,k-1] for k in ks],n90,n90/N])
    valid=np.repeat(positive[:,None],len(METRICS),axis=1);valid[:,0]=True
    if N==1:valid[:,2]=False
    values[~valid]=0
    # No smoothing or clipping; bounds include only FP64 rounding tolerance.
    for i in [2,4,5,6,7,8,10]:
        if ((values[valid[:,i],i]<-1e-12)|(values[valid[:,i],i]>1+1e-12)).any():raise ValueError('Metric range')
    if np.any(H[positive]<-1e-12) or np.any(H[positive]>math.log(N)+1e-12):raise ValueError('Entropy range')
    raw=x.mean(0); conditional=p[positive].mean(0) if positive.any() else None
    pooled=raw/M.mean() if M.mean()>0 else None
    return dict(values=values,valid=valid,positive=positive,low=low,N=N,ks=ks,raw=raw,conditional=conditional,pooled=pooled)

def scalar_rows(result, base):
    out=dict(base); v=result['values']; valid=result['valid']; positive=result['positive']; low=result['low']
    out.update(n_positions_total=len(positive),zero_mass_fraction=float((~positive).mean()),low_mass_fraction=float(low.mean()),
        low_mass_fraction_positive=float(low.sum()/positive.sum()) if positive.any() else None,
        n_positions_positive=int(positive.sum()),n_positions_low_mass=int(low.sum()),support_N=result['N'],
        top3_k=result['ks'][1],top5_k=result['ks'][2],top10pct_k=result['ks'][3])
    for i,name in enumerate(METRICS):
        mask=valid[:,i]; sens=mask & positive & ~low
        out[name]=float(v[mask,i].mean()) if mask.any() else None
        out[name+'_null_reason']=None if mask.any() else ('single_element_support' if name=='normalized_entropy' and result['N']==1 else 'zero_support_mass')
        out[name+'_n_valid']=int(mask.sum());out[name+'_sensitivity_n_valid']=int(sens.sum())
        out[name+'_sensitivity']=float(v[sens,i].mean()) if sens.any() else None
    return out

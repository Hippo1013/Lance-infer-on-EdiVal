#!/usr/bin/env python3
"""Raw-record checks of frozen windows: low-mass and total amount weighting.

This adds exact within-window pooled ratios without combining Query deviations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
from pathlib import Path
import platform
import numpy as np
from run import ROOT,HOSTS,atomic_json,sha,now


def average(x,mask):return float(x[mask].mean()) if mask.any() else None


def worker(item):
    candidates=json.loads((ROOT/'metadata/frozen_candidates.json').read_text())['candidates']
    rows=[]
    with np.load(item['original_path'],allow_pickle=False) as z:
        names=z['group_names'].tolist();stat=z['stat_names'].tolist();raw=z['group_stats'][...,stat.index('mean')].astype(float)
        region=z['region_mean'].astype(float);markers=z['image_marker_mean'].astype(float)
        for c in candidates:
            key=c['metric'];r=(slice(c['step_start'],c['step_stop']),slice(c['layer_start'],c['layer_stop']),slice(None))
            if key=='pair_T1_T2:older_share':
                a=raw[...,names.index('T1')];b=raw[...,names.index('T2')];den=a+b;value=np.divide(a,den,out=np.zeros_like(a),where=den>0);amount=den
                da=a/z['group_token_counts'][names.index('T1')];db=b/z['group_token_counts'][names.index('T2')]
                density=np.divide(da,da+db,out=np.zeros_like(a),where=da+db>0)
            elif key=='pair_I1_combined_I2_combined:older_share':
                a=raw[...,names.index('I1_vit')]+raw[...,names.index('I1_vae')];b=raw[...,names.index('I2_vit')]+raw[...,names.index('I2_vae')];den=a+b;value=np.divide(a,den,out=np.zeros_like(a),where=den>0);amount=den;density=value
            elif key=='I0_vit:left_share':
                x=region[...,0,0,:,:];a=x[...,0,0];den=x.sum(axis=(-1,-2));value=np.divide(a,den,out=np.zeros_like(a),where=den>0);amount=den;density=value
            else:
                a=markers[...,0,1,1];den=np.ones_like(a);value=a;amount=den;density=value
            vr=value[r];dr=den[r];ar=a[r];valid=dr>0;sens=dr>1e-6
            pooled=float(ar.sum()/dr.sum()) if dr.sum()>0 else None
            # The total amount assigned to two objects or a picture is separate
            # from its internal conditional shape.
            row=dict(session_id=item['session_id'],candidate_id=c['id'],metric=key,
                raw_record_equal=average(vr,valid),raw_record_sensitivity=average(vr,sens),window_amount_weighted=pooled,
                density_share=average(density[r],valid),absolute_numerator=float(ar.mean()),absolute_denominator=float(dr.mean()),
                n_records=int(vr.size),positive_records=int(valid.sum()),sensitivity_records=int(sens.sum()),
                full_record_equal=average(value,den>0),full_record_sensitivity=average(value,den>1e-6),
                full_amount_weighted=float(a.sum()/den.sum()) if den.sum()>0 else None)
            rows.append(row)
    return rows


def main(host):
    assert platform.node()==HOSTS[host]
    with gzip.open(ROOT/'metadata/source_manifest.jsonl.gz','rt') as f:items=[json.loads(x) for x in f]
    items=[x for x in items if x['host']==host and x['generation_turn']==3]
    with ProcessPoolExecutor(max_workers=6) as pool:rows=[r for batch in pool.map(worker,items) for r in batch]
    atomic_json(ROOT/'metadata'/f'{host}_window_review.json',dict(status='passed',host=host,source_candidates_sha256=sha(ROOT/'metadata/frozen_candidates.json'),rows=rows,at=now()))
    print(host,'raw window reviews',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',choices=list(HOSTS),required=True);main(p.parse_args().host)

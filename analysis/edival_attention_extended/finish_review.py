#!/usr/bin/env python3
"""Candidate variants, length strata, stable heads and machine-readable findings."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregate import describe,clean
from core import mean_valid
from run import ROOT,sha,atomic_json,now
from transfer import download


def main():
    peer=ROOT/'metadata/a800_1_window_review.json'
    if not peer.exists():
        state=json.loads((ROOT/'transfer_state.json').read_text());token=(Path(state['temp'])/'token').read_text().strip()
        download('http://172.17.61.60:18792','metadata/a800_1_window_review.json',token,peer)
    reviews=[]
    for host in ['a800_0','a800_1']:
        j=json.loads((ROOT/'metadata'/f'{host}_window_review.json').read_text());assert j['status']=='passed' and j['source_candidates_sha256']==sha(ROOT/'metadata/frozen_candidates.json');reviews.extend(j['rows'])
    assert len(reviews)==572*4
    design=json.loads((ROOT/'metadata/design.json').read_text());splits=design['split'];candidates=json.loads((ROOT/'summaries/candidate_review.json').read_text())
    lengths={}
    for sid in splits:
        meta=json.loads((ROOT/'objects'/sid/'turn_3.json').read_text());lengths[sid]=len(meta['tokens']['T1']['ids'])
    q1,q2=np.quantile(list(lengths.values()),[1/3,2/3]);strata={s:('short' if n<=q1 else 'medium' if n<=q2 else 'long') for s,n in lengths.items()}
    rows=[]
    for c in candidates['cases']:
        rr=[r for r in reviews if r['candidate_id']==c['id']]
        for split in ['exploration','confirmation','all']:
            use=[r for r in rr if split=='all' or splits[r['session_id']]['split']==split];clusters=[splits[r['session_id']]['cluster'] for r in use]
            variants={}
            for metric in ['raw_record_equal','raw_record_sensitivity','window_amount_weighted','density_share','absolute_numerator','absolute_denominator','full_record_equal','full_record_sensitivity','full_amount_weighted']:
                vals=[r[metric] if r[metric] is not None else np.nan for r in use]
                variants[metric]=describe(vals,clusters)
            stratified=[]
            for label in ['short','medium','long']:
                sub=[r for r in use if strata[r['session_id']]==label]
                subclusters=[splits[r['session_id']]['cluster'] for r in sub]
                stratified.append(dict(stratum=label,n=len(sub),record_equal=float(mean_valid([r['raw_record_equal'] for r in sub])),amount_weighted=float(mean_valid([r['window_amount_weighted'] for r in sub])),
                    record_equal_summary=describe([r['raw_record_equal'] for r in sub],subclusters),
                    amount_weighted_summary=describe([r['window_amount_weighted'] for r in sub],subclusters),
                    density_share=float(mean_valid([r['density_share'] for r in sub]))))
            rows.append(dict(candidate_id=c['id'],split=split,variants=variants,strata=stratified,
                low_record_fraction=float(np.mean([1-r['sensitivity_records']/r['n_records'] for r in use]))))
    atomic_json(ROOT/'summaries/window_variants.json',clean(dict(status='passed',rows=rows,length_cutpoints=[q1,q2],length_stratification='instruction1 actual token count; fixed terciles, ties retained')))
    # Exact flat-record mean follows the accepted first-three-stage convention.
    # Retain the position/head mean: zero denominators change implicit weights.
    for row in candidates['rows']:
        variant=next(v for v in rows if v['candidate_id']==row['id'] and v['split']==row['split'])
        if 'position_equal_summary' not in row:row['position_equal_summary']=row['summary']
        row['summary']=variant['variants']['raw_record_equal']
        row['mean_convention']='equal valid raw records within selected window; session equal'
        rr=[r for r in reviews if r['candidate_id']==row['id'] and (row['split']=='all' or splits[r['session_id']]['split']==row['split'])]
        d=np.array([r['raw_record_equal'] for r in rr])-row['reference'];tol=row['coverage_threshold']
        row['coverage']={'lower':float((d < -tol).mean()),'near':float((abs(d)<=tol).mean()),'higher':float((d>tol).mean())}
    for c in candidates['cases']:
        rr=sorted([r for r in reviews if r['candidate_id']==c['id']],key=lambda r:r['session_id'])
        values=np.array([r['raw_record_equal'] for r in rr]);ids=[r['session_id'] for r in rr]
        typical=int(np.argmin(abs(values-np.median(values))))
        strong=int(np.argmin(values) if c['direction']=='lower' else np.argmax(values))
        reverse=np.flatnonzero((values>=c['reference']) if c['direction']=='lower' else (values<=c['reference']))
        opposite=int(reverse[np.argmax(values[reverse]) if c['direction']=='lower' else np.argmin(values[reverse])]) if len(reverse) else None
        c['cases']={'typical':ids[typical],'strong':ids[strong],'opposite':ids[opposite] if opposite is not None else None}
        c['case_selection_metric']='exact window raw-record-equal ratio; median, strong and reference-reversed'
    atomic_json(ROOT/'summaries/candidate_review.json',clean(candidates))
    # A fixed head can dominate one layer; per-session top-head identity
    # frequencies should be distinguished from per-session re-ranking shares.
    stable=[]
    rates=json.loads((ROOT/'summaries/fixed_head_win_rates.json').read_text())
    for g in ['T1','T3','I0_vit','I0_vae']:
        candidates_head=[(v,k) for k,v in rates.items() if k.startswith(f't3__{g}:top_head_identity__')]
        if candidates_head:
            rate,key=max(candidates_head);stable.append(dict(group=g,fixed_layer=int(key.split('__l')[1].split('__h')[0])+1,fixed_head=int(key.split('__h')[1])+1,
                mean_win_rate=rate,note='descriptive strongest fixed head over all sessions and steps; not a holdout-selected claim'))
    with np.load(ROOT/'summaries/session_scalars.npz') as z:
        cols=z['metric_names'].tolist();values=z['values'];sids=z['session_ids'].tolist()
        trends=[]
        for key in ['paired__1_3__T1:delta_mass','paired__1_3__T1:delta_entropy','paired__1_3__T1:TV','paired__1_3__T1:top_overlap',
                    'paired__1_3__I0_vit:TV','paired__1_3__I0_vae:TV','paired__2_3__I1_vit:TV','paired__2_3__I1_vae:TV']:
            x=values[:,cols.index(key)];valid=x[np.isfinite(x)];tol=.0001 if key.endswith('mass') else .01
            trends.append(dict(metric=key,mean=float(valid.mean()),median=float(np.median(valid)),p10=float(np.quantile(valid,.1)),p90=float(np.quantile(valid,.9)),
                positive=float((valid>tol).mean()),negative=float((valid < -tol).mean()),near=float((abs(valid)<=tol).mean()),threshold=tol,n=len(valid)))
        correlations=[]
        for mod in ['vit','vae']:
            a=values[:,cols.index(f't3__I0_{mod}:left_share')];b=values[:,cols.index(f't3__I0_{mod}:marker_end_mass')];valid=np.isfinite(a)&np.isfinite(b)
            correlations.append(dict(modality=mod,session_correlation=float(np.corrcoef(a[valid],b[valid])[0,1]),n=int(valid.sum()),
                note='session-level descriptive correlation; different denominators, no competition/causal inference'))
    result=dict(status='passed',stable_heads=stable,paired_trends=trends,spatial_marker_correlations=correlations,
        split_sessions={label:sum(s['split']==label for s in splits.values()) for label in ['exploration','confirmation']},at=now())
    atomic_json(ROOT/'summaries/phenomena_facts.json',clean(result))
    print('window weighting, sensitivity, strata and phenomenon facts complete')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""CPU-only first-edition aggregation with immutable automatic/human provenance."""
import argparse, collections, copy, hashlib, json, math
from pathlib import Path

METRICS = {'IF': 'IF_qwen', 'CC': 'CC', 'GA': 'GA_qwen', 'GA_prefix': 'GA_prefix_qwen'}
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def aggregate(rows,metric):
    values=[r['effective'][metric]['score'] for r in rows if r['effective'][metric]['status']=='ok']
    return {'mean':sum(values)/len(values) if values else None,'sum':sum(values),'valid':len(values),'total':len(rows),
            'statuses':dict(collections.Counter(r['effective'][metric]['status'] for r in rows)),
            'origins':dict(collections.Counter(r['effective'][metric].get('origin','automatic') for r in rows))}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    src=args.source;out=args.output;h=read(out/'human_review_snapshot.json');proof=read(out/'source_integrity.json');pending=read(src/'pending_review.json');old=read(src/'scores.json');baseline=read(src/'summary.json')
    checks={}
    def check(name,ok):
        if not ok: raise ValueError(name)
        checks[name]=True
    check('complete_validated_automatic_source', proof['validation_status']=='passed' and proof['completion_exit_code']==0 and baseline['status']=='complete')
    check('source_review_fingerprint',h['score_fingerprint']==proof['fingerprint']==baseline['fingerprint'])
    check('review_source_hashes',all(proof['source_hashes'][n]==v for n,v in h['source_hashes'].items()))
    check('local_remote_source_hashes',all(sha(src/n)==v for n,v in proof['source_hashes'].items() if (src/n).exists()))
    keys=lambda r:(r['session_id'],r['turn'],r['metric'])
    targets={keys(i):i for i in pending['items']};ratings={keys(r):r for r in h['ratings']}
    check('exact_pending_coverage',len(targets)==len(pending['items'])==len(ratings)==len(h['ratings'])==31 and set(targets)==set(ratings))
    check('all_human_binary_resolved',all(r['decision'] in ('pass','fail') and type(r['human_score']) is int and r['human_score']==int(r['decision']=='pass') and r['version']>=1 for r in ratings.values()))
    check('human_origin_identity',all(r['judge']=='qwen' and r['vote']==1 and r['status']==targets[k]['status'] and r['cause']==targets[k]['cause'] for k,r in ratings.items()))
    check('unique_turn_records',len(old)==2160 and len({(r['session_id'],r['turn']) for r in old})==2160)
    rows=[];applied=[]
    for r in old:
        effective={m:copy.deepcopy(r[k])|{'origin':'automatic'} for m,k in METRICS.items()}
        overrides=[]
        for metric in ('IF','GA_prefix'):
            k=(r['session_id'],r['turn'],metric)
            if k in ratings:
                human=ratings[k];check('only_unresolved_overrides_'+human['id'],effective[metric]['score'] is None and effective[metric]['status'] in ('pending_review','response_incomplete'))
                effective[metric]={'status':'ok','score':human['human_score'],'origin':'human_review','human_id':human['id'],'human_version':human['version'],'review_fingerprint':h['review_fingerprint']}
                overrides.append(human['id']);applied.append(k)
        rows.append({'session_id':r['session_id'],'split':r['split'],'turn':r['turn'],'task_type':r['task_type'],'instruction':r['instruction'],
                     'automatic':{m:copy.deepcopy(r[k]) for m,k in METRICS.items()},'effective':effective,'human_override_ids':overrides})
    check('every_override_applied_once',set(applied)==set(ratings) and len(applied)==len(ratings))
    sessions=collections.defaultdict(list)
    for r in rows:sessions[r['session_id']].append(r)
    check('720_three_turn_sessions',len(sessions)==720 and all(sorted(r['turn'] for r in rs)==[1,2,3] for rs in sessions.values()))
    for sid,rs in sessions.items():
        minimum=1.0;human_prefixes=[]
        for r in sorted(rs,key=lambda r:r['turn']):
            prefix=r['effective']['GA_prefix'];check('resolved_prefix_'+sid+'_'+str(r['turn']),prefix['status']=='ok' and prefix['score'] in (0,1))
            minimum=min(minimum,prefix['score'])
            if prefix['origin']=='human_review':human_prefixes.append(prefix['human_id'])
            r['effective']['GA']={'status':'ok','score':minimum,'origin':'derived_cumulative','human_dependency_ids':list(human_prefixes)}
    check('legal_automatic_IF_prefix_and_CC_unchanged',all(r['effective'][m]['score']==r['automatic'][m]['score'] and r['effective'][m]['status']==r['automatic'][m]['status'] for r in rows for m in ('IF','CC','GA_prefix') if r['automatic'][m]['status']=='ok'))
    check('legal_automatic_cumulative_unchanged',all(r['effective']['GA']['score']==r['automatic']['GA']['score'] for r in rows if r['automatic']['GA']['status']=='ok'))
    recovered=[{'session_id':r['session_id'],'turn':r['turn'],'score':r['effective']['GA']['score']} for r in rows if r['automatic']['GA']['score'] is None]
    check('37_cumulative_dependencies_recovered',len(recovered)==37 and {(r['session_id'],r['turn']) for r in recovered}=={(r['session_id'],r['turn']) for r in h['cumulative_dependencies']})
    groups={}
    for split in ('all','cm','cu'):
        for turn in ('all',1,2,3):
            selected=[r for r in rows if (split=='all' or r['split']==split) and (turn=='all' or r['turn']==turn)]
            groups[f'{split}/turn_{turn}']={m:aggregate(selected,m) for m in METRICS}
    task_groups={t:{m:aggregate([r for r in rows if r['task_type']==t],m) for m in METRICS} for t in sorted({r['task_type'] for r in rows})}
    def session_stats(split):
        selected=[sorted(rs,key=lambda r:r['turn']) for rs in sessions.values() if split=='all' or rs[0]['split']==split]
        passed=sum(rs[-1]['effective']['GA']['score']==1 for rs in selected)
        first_failure=collections.Counter(next((r['turn'] for r in rs if r['effective']['GA']['score']==0),'none') for rs in selected)
        return {'all_rounds_GA_pass':passed,'total':len(selected),'rate':passed/len(selected),'first_failure_round':{str(k):v for k,v in first_failure.items()}}
    changes={m:{'automatic':baseline['groups']['all/turn_all'][METRICS[m]],'human_completed':groups['all/turn_all'][m],'mean_delta':groups['all/turn_all'][m]['mean']-baseline['groups']['all/turn_all'][METRICS[m]]['mean']} for m in METRICS}
    summary={'status':'complete','protocol':'mice-first-edition-qwen-once-human-v1','scope':'720 sessions/2160 turns; Qwen single-vote plus human-only resolution of 31 pending components',
             'source_inference':'outputs/mice/full_bare_20261004_attention','automatic_protocol':baseline['protocol'],'automatic_fingerprint':proof['fingerprint'],'human_review_fingerprint':h['review_fingerprint'],'human_snapshot_exported_at':h['exported_at'],
             'sessions':720,'turns':2160,'human_review':h['summary']|{'by_metric':{m:dict(collections.Counter(r['decision'] for r in ratings.values() if r['metric']==m)) for m in ('IF','GA_prefix')}},
             'groups':groups,'task_groups':task_groups,'session_results':{s:session_stats(s) for s in ('all','cm','cu')},'comparison_with_automatic':changes,
             'remaining_exclusions':{m:[{'session_id':r['session_id'],'turn':r['turn'],'status':r['effective'][m]['status'],'reason':r['effective'][m].get('reason')} for r in rows if r['effective'][m]['status']!='ok'] for m in ('IF','CC')},
             'limitations':['Project replacement-judge evaluation, not an official leaderboard score','Human ratings cover only 31 output failures, not all model judgments','Human review page exposed original judge responses; these are review labels, not a blinded accuracy calibration','No new GPU inference or sampling; no composite IF/CC/GA total']}
    check('final_effective_denominators',[(groups['all/turn_all'][m]['valid']) for m in METRICS]==[2153,1883,2160,2160])
    check('CC_unchanged',groups['all/turn_all']['CC']['mean']==baseline['groups']['all/turn_all']['CC']['mean'])
    write(out/'scores.json',rows);write(out/'summary.json',summary);write(out/'recovered_cumulative_ga.json',recovered)
    write(out/'validation.json',{'status':'passed','checks':checks,'source_integrity':proof,'artifact_hashes':{n:sha(out/n) for n in ['human_review_snapshot.json','source_integrity.json','scores.json','summary.json','recovered_cumulative_ga.json']},'gpu_calls':0})
    print(json.dumps({'overall':groups['all/turn_all'],'sessions':summary['session_results'],'human_review':summary['human_review']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

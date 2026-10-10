#!/usr/bin/env python3
"""Close this task's temporary transfer and bind final scientific/report evidence."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import time
from run import ROOT,PROJECT,HOSTS,TASK,sha,atomic_json,now


def close(host):
    assert platform.node()==HOSTS[host]
    state=json.loads((ROOT/'transfer_state.json').read_text());pid=int(state['pid']);proc=Path('/proc')/str(pid)
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
        if 'edival_attention_extended/transfer.py' not in cmd or str(state['port']) not in cmd:raise ValueError('Transfer PID was reused')
        os.kill(pid,signal.SIGTERM)
        for _ in range(30):
            stat=(proc/'stat').read_text() if proc.exists() else ''
            if not stat or stat.split()[2]=='Z':break
            time.sleep(.1)
    with socket.socket() as s:
        s.settimeout(1);assert s.connect_ex((state['ip'],state['port']))!=0
    temp=Path(state['temp']);assert str(temp).startswith('/tmp/tmp.') and not temp.is_symlink()
    if temp.exists():shutil.rmtree(temp)
    cache=PROJECT/'analysis/edival_attention_extended/__pycache__'
    if cache.exists():shutil.rmtree(cache)
    processes=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ').strip()
            if 'llamafactory_burn.sh' in args and args.split()[-1] in ['0','1']:
                processes.append(dict(pid=int(p.name),gpu=int(args.split()[-1]),loop=True,script='host-local independent burn' if host=='a800_1' else 'existing original burn'))
        except (OSError,IndexError):continue
    assert len(processes)==2 and sorted(x['gpu'] for x in processes)==[0,1],processes
    active=[]
    scripts=['run.py','merge.py','aggregate.py','plot.py','review_windows.py','check_real.py','finish_review.py','validate_publication.py']
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(any(arg.endswith('edival_attention_extended/'+name) for name in scripts) for arg in args):active.append(int(p.name))
        except OSError:continue
    assert not active,active
    state.update(closed=True,closed_at=now(),temporary_directory_removed=True)
    atomic_json(ROOT/'transfer_state.json',state)
    result=dict(status='passed',host=host,hostname=platform.node(),at=now(),transfer_closed=True,temporary_directory_removed=True,
        temporary_directory=str(temp),active_task_processes=active,burn_loops=processes,gpu_management='CPU-only analysis; burn and monitoring configuration unchanged')
    atomic_json(ROOT/'validation'/f'{host}_resources.json',result)
    print(json.dumps({k:result[k] for k in ['status','host','transfer_closed','temporary_directory_removed','active_task_processes']}))


def bind():
    assert platform.node()==HOSTS['a800_0']
    science=json.loads((ROOT/'validation/science.json').read_text());assert science['status']=='passed' and science['turns']==1716
    for host in HOSTS:
        r=json.loads((ROOT/'validation'/f'{host}_resources.json').read_text());assert r['status']=='passed' and r['transfer_closed'] and r['temporary_directory_removed']
    report=PROJECT/'outputs/attention_reports'/TASK
    fm=json.loads((report/'figure_manifest.json').read_text());assert len(fm['figures'])==28
    for f in fm['figures']:
        p=report/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes']
    local=json.loads((ROOT/'validation/local_publication.json').read_text());assert local['chart_files']==28 and local['markdown_images']==26 and local['visual_review']=='passed'
    rp=report/'attention分析结果.md';assert sha(rp)==local['report_sha256']
    accepted_manifest=json.loads((ROOT/'artifact_manifest.json').read_text())
    for f in accepted_manifest['files']:
        p=ROOT/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes']
    code={p.name:sha(p) for p in sorted((PROJECT/'analysis/edival_attention_extended').glob('*')) if p.is_file()}
    binding=dict(task_id=TASK,at=now(),status='passed',source_design_sha256=sha(ROOT/'metadata/design.json'),
        scientific_manifest_sha256=sha(ROOT/'artifact_manifest.json'),scientific_validation_sha256=sha(ROOT/'validation/science.json'),
        frozen_candidates_sha256=sha(ROOT/'metadata/frozen_candidates.json'),figure_manifest_sha256=sha(report/'figure_manifest.json'),
        report_sha256=sha(rp),local_publication_sha256=sha(ROOT/'validation/local_publication.json'),
        resources={h:sha(ROOT/'validation'/f'{h}_resources.json') for h in HOSTS},code=code)
    atomic_json(ROOT/'validation/final_binding.json',binding)
    atomic_json(ROOT/'completion.json',dict(task_id=TASK,status='passed',finished_at=now(),input_sessions=572,input_turns=1716,
        paired_objects=6864,independently_checked_turns=18,independent_paired_checks=12,baseline_scalar_comparisons=61776,
        chart_pairs=14,total_markdown_images=26,source_task='edival_semantic_20261009_v2',
        scientific_manifest_sha256=binding['scientific_manifest_sha256'],final_binding_sha256=sha(ROOT/'validation/final_binding.json'),
        report_root=str(report),report_sha256=binding['report_sha256'],resources_passed=True,temporary_directories_removed=True,
        outstanding_issues=[],verification='independent formulas and source/baseline/coverage/weight checks; local visual review; no external agent review claimed'))
    print(json.dumps({'status':'passed','sessions':572,'turns':1716,'chart_pairs':14,'report_sha256':binding['report_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['close','bind']);p.add_argument('--host',choices=list(HOSTS));a=p.parse_args()
    if a.stage=='close':close(a.host)
    else:bind()

#!/usr/bin/env python3
"""Create an isolated judge worker sharing only immutable completed metric evidence.

Optional: lets Qwen and Gemma run on separate GPUs without sharing stage locks.
No scores are copied or merged numerically. The report reads the worker's results
through a stable symlink, under the same manifest fingerprint.
"""
import argparse
import fcntl
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lance_mice.scoring.core import read,source_pins,JUDGES,digest
from lance_mice.scoring.runner import saved,artifact_path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--judge',choices=list(JUDGES),required=True);a=p.parse_args()
out=a.output.resolve()
with (out/'.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(out/'manifest.json');fp=m.pop('fingerprint')
    if digest(m)!=fp or m['sources']!=source_pins():raise ValueError('Manifest identity changed')
    if read(out/'status_metrics.json')['status']!='complete':raise ValueError('Shared metrics must be complete')
    for row in m['records']:
        for turn in range(1,4):
            if saved(artifact_path(out,'metrics',row,turn),fp) is None:raise ValueError('Incomplete shared evidence')
    worker=out/('worker_'+a.judge);destination=out/a.judge
    if worker.exists() or destination.exists() or destination.is_symlink():raise ValueError('Worker or judge result already exists')
    worker.mkdir()
    (out/'requests').mkdir(exist_ok=True)
    for name in ['manifest.json','preflight.json','metrics','detections','requests']:
        (worker/name).symlink_to(Path('..')/name,target_is_directory=name in ['metrics','detections','requests'])
    destination.symlink_to(Path(worker.name)/a.judge,target_is_directory=True)
    print(worker)

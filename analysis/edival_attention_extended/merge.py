#!/usr/bin/env python3
"""Retrieve peer derived results directly on the archive server, bound by hashes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import urllib.request
from run import ROOT,sha,atomic_json,now
from transfer import download


def main():
    state=json.loads((ROOT/'transfer_state.json').read_text());token=(Path(state['temp'])/'token').read_text().strip();base='http://172.17.61.60:18792'
    def remote_json(route):
        req=urllib.request.Request(base+'/'+route,headers={'Authorization':'Bearer '+token,'X-Task-ID':'edival_extended_20261010_v1'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=30) as r:return json.load(r)
    status=remote_json('a800_1_status.json')
    if status['stage']!='computed' or status['sessions_done']!=293 or status['turns_done']!=879:raise ValueError('Peer incomplete')
    manifest=remote_json('a800_1_manifest.json')
    atomic_json(ROOT/'metadata/a800_1_original_manifest.json',manifest);atomic_json(ROOT/'metadata/a800_1_original_status.json',status)
    def get(x):
        target=ROOT/x['path']
        if not target.is_file():download(base,x['path'],token,target)
        if target.stat().st_size!=x['bytes'] or sha(target)!=x['sha256']:raise ValueError(('Peer hash mismatch',x['path']))
        return 1
    done=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fut in as_completed([pool.submit(get,x) for x in manifest['files']]):
            done+=fut.result()
            if done%100==0:atomic_json(ROOT/'merge_status.json',dict(stage='transferring',files_done=done,files_total=len(manifest['files']),at=now()))
    download(base,'metadata/a800_1_independent_check.json',token,ROOT/'metadata/a800_1_independent_check.json')
    check=json.loads((ROOT/'metadata/a800_1_independent_check.json').read_text());assert check['status']=='passed' and check['turns']==9
    atomic_json(ROOT/'merge_status.json',dict(stage='merged',files_done=done,at=now(),independent_check='passed'))
    print('peer derived files verified',done)

if __name__=='__main__':main()

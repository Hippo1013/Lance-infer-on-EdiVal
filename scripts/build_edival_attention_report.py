#!/usr/bin/env python3
"""Wait for scientific validation, render all figures, then draft the local report."""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
ROOT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal');TASK='edival_semantic_20261009_v2'
SOURCE=ROOT/'outputs/attention_analysis'/TASK;OUT=ROOT/'outputs/attention_reports'/TASK
PYTHON='/home/chs/conda/envs/edival-attention-figures/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(stage,**kw):
    p=OUT/'status.json';q=p.with_suffix('.part');q.write_text(json.dumps(dict(stage=stage,updated=time.time(),**kw),indent=2)+'\n');q.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=True);status('waiting_for_validated_analysis')
    while not (SOURCE/'completion.json').exists():
        if (SOURCE/'validation/analysis_failure.json').exists():raise ValueError('Analysis failed; report not generated')
        time.sleep(30)
    c=json.loads((SOURCE/'completion.json').read_text());assert c['status']=='passed'
    assert sha(SOURCE/c['artifact_manifest_path'])==c['artifact_manifest_sha256']
    with tempfile.TemporaryDirectory(prefix='edival-semantic-figures-') as temp:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=temp,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for script in ['plot_edival_attention_semantic.py','plot_edival_semantic_checks.py']:
            status('rendering',script=script)
            subprocess.run([PYTHON,'-B',str(ROOT/'scripts'/script),'--input',str(SOURCE),'--output',str(OUT),'--font','/home/chs/tools/fonts/NotoSansCJKsc-Regular.otf'],env=env,cwd=temp,check=True)
        subprocess.run([PYTHON,'-B',str(ROOT/'scripts/write_edival_attention_results.py'),'--input',str(SOURCE),'--figures',str(OUT),'--output',str(OUT/'attention分析结果.md')],env=env,cwd=temp,check=True)
    # This is readiness for independent local inspection, never final acceptance.
    figures=sorted(OUT.glob('*.png'))+sorted(OUT.glob('*.svg'));assert len(figures)==24
    assert sha(SOURCE/c['artifact_manifest_path'])==c['artifact_manifest_sha256']
    readiness={'status':'ready_for_local_review','task_id':TASK,'source_artifact_manifest_sha256':c['artifact_manifest_sha256'],'report_sha256':sha(OUT/'attention分析结果.md'),
        'figures':{p.name:sha(p) for p in figures},'temporary_directory_removed':True,'updated':time.time()}
    (OUT/'publication_ready.json').write_text(json.dumps(readiness,indent=2)+'\n');status('ready_for_local_review',figures=24)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise

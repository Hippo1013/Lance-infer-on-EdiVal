#!/usr/bin/env python3
"""Create an isolated observation runtime without changing the frozen six-run runtime."""
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
NAME='edival_semantic_20261009_v2_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    frozen=ROOT/'runtime/sixrun_20261007_tokenregion_v3';runtime=ROOT/'runtime'/NAME
    expected={'attention_regions.py':'d1236ae2339edefba0c8b94a5e50efdf0cb905ebc31d5f2a2b0ec3e0543a9291','omni_pipeline.py':'8c02bc65c656a69611b39f09c6b4f7540cbfe1f7c55c9c8d88ad6096482ad799','attention.py':'58f8589e49b305ca7ef84538ff9921f58df30d198cb89231de92007b263a136f'}
    for name,h in expected.items():
        if sha(frozen/'src/lance_mice'/name)!=h:raise ValueError('Frozen source differs: '+name)
    if runtime.exists():
        if (runtime/'recollection_manifest.json').exists():raise FileExistsError(runtime)
        for name,h in expected.items():
            if sha(runtime/'src/lance_mice'/name)!=h:raise ValueError('Partial runtime has unknown edits')
    else:shutil.copytree(frozen,runtime)
    for name in ['attention_regions_v2.py','cache_semantics.py','semantic_pipeline.py']:
        shutil.copy2(ROOT/'src/lance_mice'/name,runtime/'src/lance_mice'/name)
    p=runtime/'src/lance_mice/settings.py';s=p.read_text().replace('"target-token-region-stats-v1"}', '"target-token-region-stats-v1", "target-token-region-stats-v2"}');p.write_text(s)
    p=runtime/'src/lance_mice/omni_pipeline.py';s=p.read_text().replace('if settings.attention_format == "target-token-region-stats-v1":', 'if settings.attention_format == "target-token-region-stats-v2":\n                from .attention_regions_v2 import RegionAttentionObserver as AttentionObserver\n            elif settings.attention_format == "target-token-region-stats-v1":');p.write_text(s)
    manifest={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}
    (runtime/'recollection_manifest.json').write_text(json.dumps({'base':str(frozen),'files':manifest,'observation_format':'target-token-region-stats-v2'},indent=2)+'\n')
    print(json.dumps({'runtime':str(runtime),'manifest_sha256':sha(runtime/'recollection_manifest.json')}))
if __name__=='__main__':main()

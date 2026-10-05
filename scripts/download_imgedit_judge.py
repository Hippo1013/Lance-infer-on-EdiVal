"""Fetch the authors' pinned ImgEdit-Judge inference files with upstream verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

REPO = 'sysuyy/ImgEdit'
REVISION = 'f8de753484a2b6bd37f135fd20d308b66b09a523'
ROOT = Path('/home/chs/model/ImgEdit-Judge-release')
EVIDENCE = Path(__file__).resolve().parents[1] / 'outputs/setup/imgedit_judge_20261004'


def save(name, value):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    p = EVIDENCE/name
    pending = p.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2)+'\n')
    pending.replace(p)


def main():
    save('status.json', {'state':'downloading', 'pid':os.getpid(), 'revision':REVISION})
    try:
        info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        files = [f for f in info.siblings if f.rfilename.startswith('ImgEdit_Judge/')
                 and Path(f.rfilename).name not in ('trainer_state.json', 'training_args.bin')]
        def fetch(f):
            p = Path(hf_hub_download(REPO, f.rfilename, repo_type='dataset', revision=REVISION, local_dir=ROOT))
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
            sha = h.hexdigest()
            if p.stat().st_size != f.size:raise ValueError(f'Size mismatch: {p.name}')
            if f.lfs and sha != f.lfs.sha256:raise ValueError(f'SHA mismatch: {p.name}')
            if p.suffix == '.safetensors':
                with safe_open(p,framework='pt',device='cpu') as sf: tensors=len(sf.keys())
            else:tensors=None
            return {'file':f.rfilename,'bytes':f.size,'sha256':sha,
                    'upstream_sha256':f.lfs.sha256 if f.lfs else None,'tensor_count':tensors}
        records=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for r in pool.map(fetch,files):
                records.append(r)
                save('status.json',{'state':'downloading','pid':os.getpid(),'verified_files':len(records),'total_files':len(files)})
        index=json.loads((ROOT/'ImgEdit_Judge/model.safetensors.index.json').read_text())
        if not all((ROOT/'ImgEdit_Judge'/f).is_file() for f in set(index['weight_map'].values())):
            raise ValueError('Missing shards')
        save('manifest.json',{'repo':REPO,'repo_type':'dataset','revision':REVISION,'model_path':str(ROOT/'ImgEdit_Judge'),'files':records})
        save('status.json',{'state':'verified','files':len(records),'bytes':sum(r['bytes'] for r in records),'finished_at':time.time()})
    except BaseException as e:
        save('status.json',{'state':'failed','error':f'{type(e).__name__}: {e}'})
        raise

if __name__ == '__main__':main()

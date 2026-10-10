"""Task-scoped paths, atomic exchange and authenticated read-only transport."""
import datetime, hashlib, json, os, pathlib, time, urllib.request

TASK = 'edival_semantic_20261009_v2'
PROJECT = pathlib.Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
ROOT = PROJECT / 'outputs/attention_analysis' / TASK
ARCHIVE = PROJECT / 'outputs/sixrun_20261007/experiment/inference/edival_chat'
ATTENTION = PROJECT / 'outputs/sixrun_20261007/edival_chat/run'
PEER = 'http://172.17.61.60:18789'
HOSTS = {'a800_0':'aibox-r61097f954fe-7d74d99d65-2zzkn', 'a800_1':'aibox-r7df77faf487-f8c468557-b9c9s'}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    with open(part, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(part, path)

def status(stage, revision=1, **kw):
    atomic_json(ROOT/'exchange/lead_status.json', dict(task_id=TASK, stage=stage, revision=revision, at=now(), session_id=os.environ.get('CODEX_THREAD_ID'), **kw))

def request(route, timeout=25):
    token = (ROOT/'dispatch/.peer_token').read_text().strip()
    req = urllib.request.Request(PEER+route, headers={'Authorization':'Bearer '+token, 'X-Task-ID':TASK})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)

def peer_json(route):
    with request(route) as r: return json.load(r)

def download(route, path, size, digest):
    for attempt in range(6):
        try:
            h=hashlib.sha256(); n=0
            with request(route) as r, open(path,'wb') as f:
                for block in iter(lambda:r.read(1024*1024), b''):
                    n+=len(block)
                    if n>size: raise ValueError('remote file exceeds receipt size')
                    h.update(block); f.write(block)
            if n!=size or h.hexdigest()!=digest: raise ValueError('remote size/hash mismatch: '+route)
            return
        except ValueError: raise
        except Exception:
            pathlib.Path(path).unlink(missing_ok=True)
            if attempt==5: raise
            time.sleep(min(2**attempt, 20))

#!/usr/bin/env python3
"""Task-scoped authenticated read-only source transfer on the private server IP."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import hashlib
import json
import re
import socket
ROOT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal');TASK='edival_semantic_20261009_v2'
COLLECT=ROOT/'outputs/attention_recollection'/TASK
TOKEN=(ROOT/'outputs/attention_analysis'/TASK/'dispatch/.peer_token').read_text().strip()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):self.send_error(405)
    def do_GET(self):self.get(False)
    def do_HEAD(self):self.get(True)
    def get(self,head):
        if self.headers.get('Authorization')!='Bearer '+TOKEN or self.headers.get('X-Task-ID')!=TASK:self.send_error(401);return
        data=None;path=None
        if self.path=='/health':data=json.dumps({'status':'running','hostname':socket.gethostname(),'task_id':TASK}).encode()
        elif self.path=='/export':
            items=[];sessions=0
            for stage in ['full_gpu0','full_gpu1']:
                p=COLLECT/stage/'completion.json'
                if not p.exists():self.send_error(409);return
                c=json.loads(p.read_text())
                if c['status']!='passed':self.send_error(409);return
                sessions+=c['sessions']
                for x in c['records']:
                    sid=x['session_id'];t=x['turn'];p=COLLECT/stage/'run'/sid.split('/')[-1]/'observed'/f'turn_{t}.json';q=p.with_name(f'turn_{t}.attention.npz')
                    items.append({'source_id':f'{sid}/turn_{t}','json':json.loads(p.read_text()),'json_path':str(p),'attention_path':str(q),'bytes':q.stat().st_size})
            data=json.dumps({'status':'passed','sessions':sessions,'turns':len(items),'items':items}).encode()
        elif (m:=re.fullmatch(r'/attention/edival/(\d+)/turn_([123])\.attention\.npz',self.path)):
            found=[COLLECT/s/'run'/m[1]/'observed'/f'turn_{m[2]}.attention.npz' for s in ['full_gpu0','full_gpu1']];found=[p for p in found if p.exists()]
            if len(found)!=1:self.send_error(404);return
            path=found[0]
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Length',str(len(data) if data is not None else path.stat().st_size));self.end_headers()
        if head:return
        try:
            if data is not None:self.wfile.write(data)
            else:
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):self.wfile.write(block)
        except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':ThreadingHTTPServer(('172.17.61.60',18789),Handler).serve_forever()

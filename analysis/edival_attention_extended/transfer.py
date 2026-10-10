#!/usr/bin/env python3
"""Temporary authenticated private-IP GET/HEAD service for this task only."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from urllib.parse import unquote
import urllib.request

TASK='edival_extended_20261010_v1'
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):self.send_error(405)
    def do_HEAD(self):self.deliver(True)
    def do_GET(self):self.deliver(False)
    def deliver(self,head):
        if self.headers.get('Authorization')!='Bearer '+self.server.token or self.headers.get('X-Task-ID')!=TASK:
            self.send_error(401);return
        raw=unquote(self.path).lstrip('/')
        if '..' in Path(raw).parts or '?' in raw or not raw:
            self.send_error(404);return
        p=(self.server.root/raw).resolve()
        allowed={'metadata','arrays','paired','objects','receipts'}
        if self.server.root not in p.parents or (Path(raw).parts[0] not in allowed and raw not in ['a800_0_status.json','a800_1_status.json','a800_0_manifest.json','a800_1_manifest.json']):
            self.send_error(404);return
        if not p.is_file() or p.name.startswith('.') or p.name.endswith('.part'):
            self.send_error(404);return
        self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
        if not head:
            try:
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(1024*1024),b''):self.wfile.write(b)
            except (BrokenPipeError,ConnectionResetError):pass


def download(base,route,token,target):
    req=urllib.request.Request(base+'/'+route,headers={'Authorization':'Bearer '+token,'X-Task-ID':TASK})
    q=target.with_name(target.name+'.part');target.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=60) as r,q.open('wb') as f:
        for b in iter(lambda:r.read(1024*1024),b''):f.write(b)
    q.replace(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--token',type=Path,required=True);p.add_argument('--bind');p.add_argument('--port',type=int,default=18791);p.add_argument('--pull');p.add_argument('--routes',nargs='*')
    a=p.parse_args();token=a.token.read_text().strip()
    if a.pull:
        for route in a.routes:download(a.pull,route,token,a.root/route)
    else:
        server=ThreadingHTTPServer((a.bind,a.port),Handler);server.root=a.root.resolve();server.token=token;server.serve_forever()

#!/usr/bin/env python3
"""Loopback-only human review of saved ImgEdit outputs; standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import shutil
import sqlite3
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT/'outputs/imgedit/full_20261004_attention/run'
DEFAULT_REVIEW = PROJECT/'outputs/review/imgedit_human_20261004'
CATEGORIES = {'content_memory':'内容记忆', 'content_understand':'上下文理解', 'version_backtrace':'版本回溯'}
DECISIONS = ('unrated', 'pass', 'fail', 'uncertain')
PROTOCOL = {
    'id':'imgedit-human-review-v1',
    'scope':'项目人工逐轮评审；官方多轮采用人工判断，但未公开可直接复现的完整表单及聚合实现。本工具的待定、备注和统计规则为项目补充。',
    'rule':'结合截至本轮的原始指令、原图和实际生成历史，判断当前结果是否完成本轮编辑，并遵守仍然有效的历史要求。遵守撤销、回溯和指代要求；不因上一轮失败就自动判本轮失败。',
    'pass':'通过：当前编辑及仍然有效的历史要求均满足。',
    'fail':'不通过：存在明确未满足的当前指令、历史约束、指代或版本要求。',
    'uncertain':'待定：图像、指令歧义或细节不足以可靠判断；暂不计入通过率。',
    'aggregation':'逐轮通过率 = 通过 / (通过 + 不通过)；未评、待定单独报告。会话全通过率只统计所有轮次均已明确判定的会话。该汇总不是官方榜单分数。',
    'source':'https://github.com/PKU-YuanGroup/ImgEdit/blob/b79848168744c8db8389086d01fe3def1758aa45/Benchmark/Multiturn/Multiturn_readme.md'
}


def read_json(path): return json.loads(path.read_text(encoding='utf-8'))

def sha256(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(args):
    run=args.run.resolve(); trans=read_json(args.translations)['sessions']
    run_manifest=read_json(run/'run.json')
    samples=run_manifest['samples']
    if len(samples)!=30 or sum(len(x[1]) for x in samples)!=88:raise ValueError('Expected full 30-session / 88-turn ImgEdit run')
    sessions=[]
    for sid, instructions in samples:
        folder=run/sid; spec=read_json(folder/'session.json'); tr=trans[sid]
        if spec['session_id']!=sid or spec['instructions']!=instructions or tr['en']!=instructions or tr['fingerprint']!=spec['fingerprint']:
            raise ValueError(f'Translation or session identity mismatch: {sid}')
        if len(tr['zh'])!=len(instructions) or not all(tr['zh']):raise ValueError('Missing translation')
        images=[]
        for turn in range(len(instructions)+1):
            name='turn_0_input.png' if turn==0 else f'turn_{turn}.png'; path=folder/name
            if not path.resolve().is_relative_to(run) or not path.is_file():raise ValueError(f'Invalid image: {path}')
            if turn:
                meta=read_json(folder/f'turn_{turn}.json')
                if meta['instruction']!=instructions[turn-1] or meta['session_fingerprint']!=spec['fingerprint'] or meta['turn']!=turn:raise ValueError('Turn metadata mismatch')
            images.append({'path':f'{sid}/{name}','url':f'/image/{sid}/{turn}','sha256':sha256(path)})
        sessions.append({'id':sid,'split':sid.split('/')[0],'fingerprint':spec['fingerprint'],
                         'instructions_en':instructions,'instructions_zh':tr['zh'],'images':images})
    manifest={'version':1,'run':str(run),'run_fingerprint':run_manifest['fingerprint'],'protocol':PROTOCOL,
              'translations_sha256':sha256(args.translations),'categories':CATEGORIES,'sessions':sessions}
    p=args.review/'manifest.json'
    if p.exists():
        if read_json(p)!=manifest:raise ValueError('Manifest changed; choose a new review directory')
    else:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'sessions':len(sessions),'turns':88,'images':sum(len(s['images']) for s in sessions),'manifest':str(p)}))


class Store:
    def __init__(self, path, manifest):
        self.path=path;self.manifest=manifest
        self.valid={(s['id'],t):s['fingerprint'] for s in manifest['sessions'] for t in range(1,len(s['instructions_en'])+1)}
        identity=hashlib.sha256(json.dumps(manifest,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT NOT NULL)')
            existing=db.execute('SELECT fingerprint FROM identity WHERE id=1').fetchone()
            if existing and existing[0]!=identity:raise ValueError('Ratings belong to another manifest')
            db.execute('INSERT OR IGNORE INTO identity VALUES (1,?)',(identity,))
            db.execute('CREATE TABLE IF NOT EXISTS ratings (session TEXT, turn INTEGER, decision TEXT NOT NULL, note TEXT NOT NULL, reviewer TEXT NOT NULL, version INTEGER NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(session,turn))')
    def connect(self):return sqlite3.connect(self.path,timeout=10)
    def rows(self):
        with self.connect() as db:
            db.row_factory=sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM ratings ORDER BY session,turn')]
    def update(self, data):
        sid=data.get('session');turn=data.get('turn')
        if not isinstance(sid,str) or type(turn)!=int or (sid,turn) not in self.valid:raise ValueError('Unknown session or turn')
        if data.get('fingerprint')!=self.valid[(sid,turn)]:raise ValueError('Session fingerprint differs')
        if data.get('decision') not in DECISIONS:raise ValueError('Invalid decision')
        if type(data.get('version'))!=int or data['version']<0:raise ValueError('Invalid version')
        if not isinstance(data.get('note'),str) or len(data['note'])>4000:raise ValueError('Invalid note')
        if not isinstance(data.get('reviewer'),str) or len(data['reviewer'])>80:raise ValueError('Invalid reviewer')
        row={k:data[k] for k in ('session','turn','decision','note','reviewer')}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT version FROM ratings WHERE session=? AND turn=?',(sid,turn)).fetchone()
            version=old[0] if old else 0
            if version!=data['version']:raise FileExistsError('此轮已被另一页面修改。请保留当前备注，刷新后核对。')
            row.update(version=version+1,updated_at=datetime.now(timezone.utc).isoformat())
            db.execute('INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?,?)',tuple(row[k] for k in ('session','turn','decision','note','reviewer','version','updated_at')))
        return row
    def export(self):
        stored={(r['session'],r['turn']):r for r in self.rows()};rows=[]
        for s in self.manifest['sessions']:
            for t,en in enumerate(s['instructions_en'],1):
                row=stored.get((s['id'],t),{'session':s['id'],'turn':t,'decision':'unrated','note':'','reviewer':'','version':0,'updated_at':None})
                rows.append({**row,'category':s['split'],'instruction_en':en,'instruction_zh':s['instructions_zh'][t-1],
                             'session_fingerprint':s['fingerprint'],'image_sha256':s['images'][t]['sha256']})
        def summary(items):
            counts={d:sum(x['decision']==d for x in items) for d in DECISIONS};n=counts['pass']+counts['fail']
            return {**counts,'total':len(items),'decided':n,'pass_rate':counts['pass']/n if n else None}
        complete=passed=0
        for s in self.manifest['sessions']:
            decisions=[r['decision'] for r in rows if r['session']==s['id']]
            if all(d in ('pass','fail') for d in decisions):
                complete+=1;passed+=all(d=='pass' for d in decisions)
        return {'protocol':self.manifest['protocol'],'run_fingerprint':self.manifest['run_fingerprint'],
                'exported_at':datetime.now(timezone.utc).isoformat(),'summary':summary(rows),
                'by_category':{c:summary([r for r in rows if r['category']==c]) for c in CATEGORIES},
                'by_turn':{str(t):summary([r for r in rows if r['turn']==t]) for t in (1,2,3)},
                'sessions':{'total':len(self.manifest['sessions']),'fully_decided':complete,'all_pass':passed,'all_pass_rate':passed/complete if complete else None},'ratings':rows}


def serve(args):
    manifest=read_json(args.review/'manifest.json');run=Path(manifest['run']).resolve();images={}
    for s in manifest['sessions']:
        for item in s['images']:
            p=(run/item['path']).resolve()
            if not p.is_relative_to(run) or sha256(p)!=item['sha256']:raise ValueError(f'Image changed: {p}')
            images[item['url']]=p
    store=Store(args.review/'ratings.sqlite3',manifest); token=secrets.token_urlsafe(32)
    public={k:v for k,v in manifest.items() if k!='run'}
    class Handler(BaseHTTPRequestHandler):
        def send(self, body, mime='application/json; charset=utf-8', status=200, filename=None):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            route=urlsplit(self.path).path
            if route=='/':self.send((PROJECT/'web/imgedit-review.html').read_bytes(),'text/html; charset=utf-8')
            elif route=='/api/init':self.send({'manifest':public,'ratings':store.rows(),'token':token})
            elif route=='/api/export.json':self.send(store.export(),filename='imgedit-human-ratings.json')
            elif route=='/api/export.csv':
                out=io.StringIO(); fields=['category','session','turn','decision','reviewer','note','instruction_zh','instruction_en','updated_at','session_fingerprint','image_sha256']
                writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader()
                for row in store.export()['ratings']:
                    # Prevent formula interpretation when opening a spreadsheet export.
                    writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v) for k,v in row.items()})
                self.send(('\ufeff'+out.getvalue()).encode(),'text/csv; charset=utf-8',filename='imgedit-human-ratings.csv')
            elif route in images:self.send(images[route].read_bytes(),'image/png')
            else:self.send({'error':'Not found'},status=404)
        def do_POST(self):
            if urlsplit(self.path).path!='/api/rating':self.send({'error':'Not found'},status=404);return
            if self.headers.get('X-Review-Token')!=token:self.send({'error':'请刷新页面后重试。'},status=403);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=32768:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(n));row=store.update(data)
                self.send({'rating':row})
            except FileExistsError as e:self.send({'error':str(e)},status=409)
            except (ValueError,TypeError,KeyError,AttributeError) as e:self.send({'error':str(e)},status=400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'ImgEdit human review: http://127.0.0.1:{args.port}; {len(images)} images',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','serve'])
    p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--review',type=Path,default=DEFAULT_REVIEW)
    p.add_argument('--translations',type=Path,default=PROJECT/'configs/imgedit_review_zh.json');p.add_argument('--port',type=int,default=8767)
    args=p.parse_args();(prepare if args.command=='prepare' else serve)(args)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent human resolution of MICE Qwen pending items; no model calls."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SCORE=ROOT/'outputs/scoring/full_qwen_once_pending_v3'
DEFAULT_REVIEW=ROOT/'outputs/review/mice_qwen_pending_v3'
RULES={
 'IF':'仅判断当前轮的指定编辑是否完成，使用上一轮图像与当前结果对照，并参考显式目标指令。完成记1，未完成记0；不要把GA的全局约束或CC的内容保持分数混入IF。文字编辑按目标文字是否正确、完整呈现判断。',
 'GA_prefix':'判断第1轮到当前轮的完整前缀。逐轮比较输入和输出，核对编辑指令及该会话的约束/指代要求。所有轮次都成功记1；任意一轮失败记0。请按下方本项的原始GA评分提示核对具体范围。',
 'CM':'首轮全局约束仅作用于当轮添加或修改的对象，不得用全图滤镜代替，也不得修改无关区域。移除类操作豁免全局约束，但仍须完成移除。',
 'CU':'按完整历史解析代词和对象指代，结合显式参照指令检查目标；不要因为孤立看当前指令不明确而猜测。',
 'saving':'无法确定可保留待定并备注；未评和待定都保持null。人工评分独立存储，原自动结果不覆盖。37轮累计GA依赖后续依据原始前缀人工结果重算，无需额外逐项评分。'
}
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(m): return hashlib.sha256(json.dumps(m,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def prepare(args):
 score=args.score.resolve();a=read(score/'pending_review.json');m=read(score/'manifest.json');v=read(score/'validation.json');c=read(score/'completion.json')
 if v['status']!='passed' or c['exit_code']!=0 or c['fingerprint']!=m['fingerprint']:raise ValueError('Unvalidated source')
 records={r['session_id']:r for r in m['records']};items=[]
 for k,item in enumerate(a['items']):
  r=records[item['session_id']];t=item['turn'];md=r['metadata'];images=[]
  for n in range(t+1):
   p=Path(r['images'][n]).resolve()
   if not p.is_relative_to(ROOT/'outputs/mice/full_bare_20261004_attention/run') or sha(p)!=r['image_sha256'][n]:raise ValueError('Source image changed')
   images.append({'path':str(p),'url':f'/image/{k}/{n}','sha256':r['image_sha256'][n],'turn':n})
  allowed=r['pixel_hashes'][:t+1]
  if any(h not in allowed for tr in item['traces'] for h in tr['image_hashes']):raise ValueError('Trace image differs')
  items.append({**item,'id':str(k),'split':r['split'],'instructions':md['instruction'][:t], 'formatted':md['formatted_instruction'][:t], 'task_type':md['task_type'][t-1], 'images':images})
 manifest={'protocol':'mice-pending-human-v1','score_fingerprint':m['fingerprint'],'source':str(score),'source_hashes':{n:sha(score/n) for n in ('pending_review.json','manifest.json','validation.json','completion.json')},'rules':RULES,'items':items,'cumulative_dependencies':a['cumulative_dependencies']}
 p=args.review/'manifest.json';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and read(p)!=manifest:raise ValueError('Existing review identity changed')
 if not p.exists():p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'items':len(items),'sessions':len({i['session_id'] for i in items}),'images':len({i['path'] for x in items for i in x['images']})}))
class Store:
 def __init__(self,path,manifest):
  self.path=path;self.m=manifest;self.ids={x['id'] for x in manifest['items']};self.fp=identity(manifest)
  with self.connect() as db:
   db.execute('CREATE TABLE IF NOT EXISTS identity (fingerprint TEXT)');row=db.execute('SELECT fingerprint FROM identity').fetchone()
   if row and row[0]!=self.fp:raise ValueError('Review identity differs')
   if not row:db.execute('INSERT INTO identity VALUES (?)',(self.fp,))
   db.execute('CREATE TABLE IF NOT EXISTS ratings (id TEXT PRIMARY KEY, decision TEXT, note TEXT, version INTEGER, updated_at TEXT)')
 def connect(self):return sqlite3.connect(self.path,timeout=10)
 def rows(self):
  with self.connect() as db:
   db.row_factory=sqlite3.Row;return [dict(r) for r in db.execute('SELECT * FROM ratings ORDER BY CAST(id AS INTEGER)')]
 def update(self,d):
  if d.get('id') not in self.ids or d.get('fingerprint')!=self.fp:raise ValueError('Unknown item or source')
  if d.get('decision') not in ('pass','fail','uncertain','unrated'):raise ValueError('Invalid decision')
  if type(d.get('version'))!=int or d['version']<0 or not isinstance(d.get('note'),str) or len(d['note'])>4000:raise ValueError('Invalid fields')
  with self.connect() as db:
   db.execute('BEGIN IMMEDIATE');old=db.execute('SELECT version FROM ratings WHERE id=?',(d['id'],)).fetchone();version=old[0] if old else 0
   if version!=d['version']:raise FileExistsError('另一页面已修改此项，请刷新后核对。')
   row={'id':d['id'],'decision':d['decision'],'note':d['note'],'version':version+1,'updated_at':datetime.now(timezone.utc).isoformat()}
   db.execute('INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)',tuple(row.values()))
  return row
 def export(self):
  rated={r['id']:r for r in self.rows()};rows=[]
  for i in self.m['items']:
   r=rated.get(i['id'],{'decision':'unrated','note':'','version':0,'updated_at':None})
   rows.append({k:i[k] for k in ('id','session_id','turn','judge','vote','metric','status','cause')}|r|{'human_score':{'pass':1,'fail':0}.get(r['decision'])})
  counts={d:sum(r['decision']==d for r in rows) for d in ('pass','fail','uncertain','unrated')}
  return {'protocol':self.m['protocol'],'score_fingerprint':self.m['score_fingerprint'],'review_fingerprint':self.fp,'source_hashes':self.m['source_hashes'],'exported_at':datetime.now(timezone.utc).isoformat(),'summary':counts|{'total':len(rows),'resolved':counts['pass']+counts['fail']},'ratings':rows,'cumulative_dependencies':self.m['cumulative_dependencies'],'application':'Independent human decisions; original automatic scores unchanged; recalculate dependent cumulative GA after resolution.'}
def serve(args):
 m=read(args.review/'manifest.json');source=Path(m['source'])
 for n,h in m['source_hashes'].items():
  if sha(source/n)!=h:raise ValueError('Scoring evidence changed')
 images={i['url']:Path(i['path']) for x in m['items'] for i in x['images']}
 for x in m['items']:
  for i in x['images']:
   if sha(Path(i['path']))!=i['sha256']:raise ValueError('Image changed')
 store=Store(args.review/'ratings.sqlite3',m);token=secrets.token_urlsafe(32);public={k:v for k,v in m.items() if k!='source'}
 public['items']=[x|{'images':[{k:v for k,v in i.items() if k!='path'} for i in x['images']]} for x in m['items']]
 class Handler(BaseHTTPRequestHandler):
  def send(self,body,mime='application/json; charset=utf-8',status=200,filename=None):
   if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
   self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
   self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'")
   if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
   self.end_headers()
   try:self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
  def do_GET(self):
   route=urlsplit(self.path).path
   if route=='/':self.send((ROOT/'web/mice-pending-review.html').read_bytes(),'text/html; charset=utf-8')
   elif route=='/api/init':self.send({'manifest':public,'ratings':store.rows(),'fingerprint':store.fp,'token':token})
   elif route=='/api/export.json':self.send(store.export(),filename='mice-qwen-human-decisions.json')
   elif route in images:self.send(images[route].read_bytes(),'image/png')
   else:self.send({'error':'Not found'},status=404)
  def do_POST(self):
   if urlsplit(self.path).path!='/api/rating':self.send({'error':'Not found'},status=404);return
   if self.headers.get('X-Review-Token')!=token:self.send({'error':'请刷新以恢复保存连接'},status=403);return
   try:
    n=int(self.headers.get('Content-Length','0'))
    if not 0<n<=32768:raise ValueError('Invalid size')
    self.send({'rating':store.update(json.loads(self.rfile.read(n)))})
   except FileExistsError as e:self.send({'error':str(e)},status=409)
   except (ValueError,TypeError,KeyError) as e:self.send({'error':str(e)},status=400)
 server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);print(f'MICE human review: http://127.0.0.1:{args.port}',flush=True)
 try:server.serve_forever()
 finally:server.server_close()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','serve']);p.add_argument('--score',type=Path,default=DEFAULT_SCORE);p.add_argument('--review',type=Path,default=DEFAULT_REVIEW);p.add_argument('--port',type=int,default=8772);a=p.parse_args();(prepare if a.command=='prepare' else serve)(a)
if __name__=='__main__':main()

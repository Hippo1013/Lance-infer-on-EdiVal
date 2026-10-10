#!/usr/bin/env python3
"""Relocate completed sixrun files without rewriting first-run evidence."""
import argparse,datetime,json,os,shutil,subprocess,tempfile
from pathlib import Path
SHARED=Path('/media/damoxing/tangzecong/chs_umm_shared')
PROJECT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal')
LOCAL=PROJECT/'outputs/sixrun_20261007/experiment'
def run(argv): subprocess.run(argv,check=True)
def copy_verified(source,dest,symlink=False):
 if dest.exists() and not dest.is_symlink():raise RuntimeError(f'Refusing existing destination {dest}')
 staging=Path(tempfile.mkdtemp(prefix='umm-storage-'))
 try:
  copied=staging/'payload';run(['cp','-a',str(source),str(copied)])
  run(['diff','-qr','--no-dereference',str(source),str(copied)])
  dest.parent.mkdir(parents=True,exist_ok=True)
  if symlink:
   assert dest.is_symlink() and dest.resolve()==source.resolve();dest.unlink()
  os.rename(copied,dest)
 finally:shutil.rmtree(staging)
def main():
 p=argparse.ArgumentParser();p.add_argument('host',choices=['a800_0','a800_1']);args=p.parse_args()
 hostname=subprocess.check_output(['hostname'],text=True).strip()
 expected={'a800_0':'aibox-r61097f954fe-7d74d99d65-2zzkn','a800_1':'aibox-r7df77faf487-f8c468557-b9c9s'};assert hostname==expected[args.host]
 evidence=PROJECT/'outputs/sixrun_20261007/storage_migration';evidence.mkdir(parents=True,exist_ok=True)
 records=[]
 def move(src,dst,link=False):
  print('COPY',src,'TO',dst,flush=True);copy_verified(src,dst,link)
  records.append({'source':str(src),'destination':str(dst),'byte_comparison':'passed','symbolic_link_replaced':link})
  (evidence/f'{args.host}.json').write_text(json.dumps({'host':args.host,'status':'in_progress','records':records},indent=2)+'\n');print('VERIFIED',dst,flush=True)
 if args.host=='a800_0':
  move(SHARED/'experiments/sixrun_20261007',LOCAL)
  move(SHARED/'setup',evidence/'resource_setup')
  move(SHARED/'sixrun_setup',evidence/'sixrun_setup')
 else:
  for name in ['lance','mice-metrics','mice-judge','EdiVal','EdiVal-judge','EdiVal-hps']:
   move(SHARED/'envs'/name,Path('/home/chs/conda/envs')/name,True)
  move(SHARED/'tools/EdiVal',Path('/home/chs/tools/EdiVal'),True)
  move(SHARED/'dataset',Path('/home/chs/dataset'),True)
  move(SHARED/'model',Path('/home/chs/model'),True)
 (evidence/f'{args.host}.json').write_text(json.dumps({'host':args.host,'hostname':hostname,'status':'passed','finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2)+'\n')
 print('MIGRATION_COPY_PASSED',args.host,flush=True)
if __name__=='__main__':main()

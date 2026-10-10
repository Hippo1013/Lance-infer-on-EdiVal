import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('review',Path(__file__).resolve().parents[1]/'scripts/mice_pending_review.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class ReviewTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='mice-human-test-');self.path=Path(self.tmp.name)/'ratings.sqlite3'
  self.m={'protocol':'test','score_fingerprint':'original','source_hashes':{'pending_review.json':'immutable'},'items':[{'id':'0','session_id':'cm/a','turn':2,'judge':'qwen','vote':1,'metric':'GA_prefix','status':'pending_review','cause':'decision_format'}],'cumulative_dependencies':[{'session_id':'cm/a','turn':3}]};self.s=r.Store(self.path,self.m)
 def tearDown(self):self.tmp.cleanup()
 def update(self,**kw):return self.s.update({'id':'0','fingerprint':self.s.fp,'decision':'pass','note':'人工确认','version':0}|kw)
 def test_persistent_scores_and_conflict(self):
  self.update();s=r.Store(self.path,self.m);self.assertEqual(s.export()['ratings'][0]['human_score'],1)
  with self.assertRaises(FileExistsError):self.update(decision='fail')
  self.update(decision='uncertain',version=1);self.assertIsNone(s.export()['ratings'][0]['human_score']);self.assertEqual(s.export()['summary']['resolved'],0)
 def test_identity_and_fields(self):
  for changes in ({'id':'x'},{'fingerprint':'other'},{'decision':'1'},{'version':True},{'note':'x'*4001}):
   with self.assertRaises(ValueError):self.update(**changes)
  with self.assertRaises(ValueError):r.Store(self.path,self.m|{'score_fingerprint':'wrong'})
 def test_export_keeps_source_and_unresolved_dependencies(self):
  out=self.s.export();self.assertEqual(out['ratings'][0]['decision'],'unrated');self.assertIsNone(out['ratings'][0]['human_score']);self.assertEqual(out['source_hashes'],self.m['source_hashes']);self.assertEqual(len(out['cumulative_dependencies']),1)
if __name__=='__main__':unittest.main()

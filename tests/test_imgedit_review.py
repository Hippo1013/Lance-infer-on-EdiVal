import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

spec=importlib.util.spec_from_file_location('imgedit_review',Path(__file__).resolve().parents[1]/'scripts/imgedit_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(subprocess.check_output(['mktemp','-d'],text=True).strip())
        self.manifest={'protocol':m.PROTOCOL,'run_fingerprint':'run-test','sessions':[{'id':'content_memory/00001','split':'content_memory','fingerprint':'case-test','instructions_en':['one','two'],'instructions_zh':['一','二'],'images':[{'sha256':str(i)} for i in range(3)]}]}
        self.store=m.Store(self.root/'ratings.sqlite3',self.manifest)
    def tearDown(self):shutil.rmtree(self.root)
    def data(self,**extra):return {'session':'content_memory/00001','turn':1,'decision':'pass','note':'测试备注','reviewer':'测试','version':0,'fingerprint':'case-test',**extra}
    def test_persistence_and_conflict(self):
        self.store.update(self.data())
        reloaded=m.Store(self.root/'ratings.sqlite3',self.manifest)
        self.assertEqual(reloaded.rows()[0]['note'],'测试备注')
        with self.assertRaises(FileExistsError):reloaded.update(self.data(decision='fail'))
        self.assertEqual(reloaded.rows()[0]['decision'],'pass')
    def test_unrated_uncertain_denominator_and_session_completion(self):
        self.assertIsNone(self.store.export()['summary']['pass_rate'])
        self.store.update(self.data());self.store.update(self.data(turn=2,decision='uncertain'))
        report=self.store.export()
        self.assertEqual(report['summary']['pass_rate'],1)
        self.assertEqual(report['summary']['uncertain'],1)
        self.assertEqual(report['sessions']['fully_decided'],0)
        self.store.update(self.data(turn=2,version=1,decision='fail'))
        report=self.store.export()
        self.assertEqual(report['summary']['pass_rate'],0.5)
        self.assertEqual(report['sessions']['fully_decided'],1)
        self.assertEqual(report['sessions']['all_pass'],0)
    def test_invalid_input_and_identity(self):
        for extra in [{'turn':3},{'turn':True},{'decision':'yes'},{'fingerprint':'other'},{'note':'x'*4001},{'version':-1}]:
            with self.assertRaises(ValueError):self.store.update(self.data(**extra))
        other=json.loads(json.dumps(self.manifest));other['run_fingerprint']='other-run'
        with self.assertRaises(ValueError):m.Store(self.root/'ratings.sqlite3',other)
        self.assertEqual(self.store.rows(),[])
    def test_clear_is_unrated_not_failure(self):
        self.store.update(self.data());self.store.update(self.data(version=1,decision='unrated',note=''))
        self.assertEqual(self.store.export()['summary']['unrated'],2)
        self.assertIsNone(self.store.export()['summary']['pass_rate'])

if __name__=='__main__':unittest.main()

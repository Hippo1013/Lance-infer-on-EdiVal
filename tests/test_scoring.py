import copy
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from lance_mice.scoring.core import parse_yes_no,parse_ga,cumulative,pair_mean,result,digest,valid_if
from lance_mice.scoring.if_rules import evaluate_single
from lance_mice.scoring.runner import infer_if,persist,saved,load_manifest
from lance_mice.scoring.core import source_pins,SETTINGS,write,request_identity,sampling_seed,vote_mean
from types import SimpleNamespace
from lance_mice.scoring.request_cache import RequestCache
from lance_mice.scoring.judge import Judge
from lance_mice.images import image_hash
from lance_mice.scoring.metrics import Detector

class ScoringTests(unittest.TestCase):
    def test_full_run_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            spec={'sources':source_pins(),'settings':SETTINGS,'selection':'all'}
            write(Path(td)/'manifest.json',{**spec,'fingerprint':digest(spec)})
            with self.assertRaisesRegex(ValueError,'Full scoring requires'):
                load_manifest(SimpleNamespace(output=Path(td),allow_full=False))
    def test_error_retry_preserves_previous_raw_evidence(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            p=Path(td)/'row.json'
            persist(p,{'fingerprint':'x','status':'error','traces':[{'raw':'ambiguous'}]})
            with self.assertRaises(ValueError):saved(p,'x')
            self.assertIsNone(saved(p,'x',retry_errors=True))
            persist(p,{'fingerprint':'x','status':'ok','traces':[{'raw':'no'}]})
            self.assertEqual(saved(p,'x')['previous_attempts'][0]['traces'][0]['raw'],'ambiguous')
    def test_raw_replay_uses_no_model_and_preserves_answer(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            image=Image.new('RGB',(32,32),'red');prompt='Answer yes or no.'
            trace={'model':'model','kind':'IF','image_hashes':[image_hash(image)],'image_sizes':[[32,32]],
                   'prompt':prompt,'request_hash':request_identity([image_hash(image)],prompt,1),
                   'vote':1,'sampling_seed':sampling_seed(1),'temperature':SETTINGS['temperature'],
                   'raw':'Therefore, the answer is no.','finish_reason':'stop','prompt_tokens':20,'completion_tokens':8}
            cache=RequestCache(td,'model',{'revision':'r'},'old-producer');cache.put(trace)
            judge=Judge('model',cache)
            def forbidden():raise AssertionError('Replay must not load a model')
            judge.load=forbidden
            self.assertEqual(judge.one(image,prompt),'no')
            self.assertEqual(judge.traces[0]['raw'],trace['raw'])
            self.assertEqual(judge.traces[0]['reused_from']['producer_fingerprint'],'old-producer')
    def test_raw_cache_rejects_conflicts_and_different_revisions(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            trace={'model':'model','image_hashes':['hash'],'prompt':'p','request_hash':request_identity(['hash'],'p',1),
                   'vote':1,'sampling_seed':sampling_seed(1),'temperature':SETTINGS['temperature'],
                   'raw':'yes','finish_reason':'stop'}
            cache=RequestCache(td,'model',{'revision':'r'},'producer');cache.put(trace)
            with self.assertRaises(ValueError):cache.put({**trace,'raw':'no'})
            with self.assertRaises(ValueError):cache.put({**trace,'finish_reason':'length'})
            self.assertIsNone(RequestCache(td,'model',{'revision':'changed'},'p').get(trace['request_hash']))
            self.assertIsNone(cache.get(request_identity(['hash'],'p',2)))
    def test_vote_average_requires_both_results(self):
        self.assertEqual(vote_mean([result('ok',1),result('ok',0)])['score'],.5)
        self.assertIsNone(vote_mean([result('ok',1),result('error')])['score'])
        with self.assertRaises(ValueError):vote_mean([result('ok',1)])
        self.assertNotEqual(sampling_seed(1),sampling_seed(2))
    def test_strict_answers(self):
        self.assertEqual(parse_yes_no('YES.'),'yes')
        self.assertEqual(parse_yes_no('The requested replacement did not occur.\n\nno'),'no')
        self.assertEqual(parse_yes_no('Explanation.\n**Yes**'),'yes')
        self.assertEqual(parse_yes_no('Reasoning.\nTherefore, the answer is no.'),'no')
        self.assertEqual(parse_yes_no('No. The subject is still present.'),'no')
        self.assertEqual(parse_yes_no('Answer: \"YES\"'),'yes')
        for x in ['yes and no','yesterday','', 'I think yes', 'yes\nno', 'The answer is not yes.', 'If that were true, the answer is yes.']:
            with self.assertRaises(ValueError):parse_yes_no(x)
    def test_ga_structure(self):
        self.assertEqual(parse_ga('<answer_turn_1>yes</answer_turn_1><answer_turn_2>no</answer_turn_2><answer_final>no</answer_final>',3),0)
        for x in ['<answer_final>yes</answer_final>', '<answer_turn_1>no</answer_turn_1><answer_final>yes</answer_final>', '<answer_turn_2>yes</answer_turn_2><answer_final>yes</answer_final>']:
            with self.assertRaises(ValueError):parse_ga(x,2)
    def test_average_after_cumulative(self):
        a=cumulative([result('ok',1),result('ok',0),result('ok',1)])
        b=cumulative([result('ok',1),result('ok',1),result('ok',1)])
        self.assertEqual([pair_mean(x,y)['score'] for x,y in zip(a,b)],[1,.5,.5])
        self.assertIsNone(pair_mean(result('ok',1),result('error'))['score'])
        self.assertIsNone(cumulative([result('error'),result('ok',1)])[1]['score'])
    def test_invalid_text_annotation(self):
        self.assertFalse(valid_if('text_change',"Add text '[]' on the image"))
        self.assertTrue(valid_if('text_change',"Add text '[hello]' on the image"))
    def test_removal_detector_failure_never_success(self):
        im=Image.new('RGB',(32,32));n=0
        def det(*a,**kw):
            nonlocal n;n+=1
            if n==2:raise RuntimeError('GPU failure')
            return {'box':[[.1,.1,.5,.5]],'score':[.9],'center':[[10,10]]}
        with self.assertRaises(RuntimeError):evaluate_single(im,im,'Remove cat','Remove [cat]','subject_remove',det,lambda *a:'no',lambda *a:'no')
    def test_dual_precheck_and_shared_detection(self):
        im=Image.new('RGB',(32,32));calls=[]
        def det(*a,**kw):calls.append(1);return {'score':[.9],'box':[[.1,.1,.5,.5]],'center':[[10,10]]}
        row={'metadata':{'instruction':['Add cat'],'task_type':['subject_add'],'formatted_instruction':['Add [cat]']}}
        a=infer_if(row,1,[im,im],det,lambda *a:'yes',lambda *a:'yes')
        b=infer_if(row,1,[im,im],det,lambda *a:'no',lambda *a:'no')
        self.assertEqual([a['score'],b['score'],len(calls)],[1,0,1])
    def test_resume_refuses_tampering(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            p=Path(td)/'row.json';persist(p,{'fingerprint':'x','status':'ok','value':1})
            self.assertEqual(saved(p,'x')['value'],1)
            with self.assertRaises(ValueError):saved(p,'other')
            p.write_text(p.read_text().replace('"value": 1','"value": 2'))
            with self.assertRaises(ValueError):saved(p,'x')
    def test_detector_cache_reuse_and_missing(self):
        with tempfile.TemporaryDirectory(prefix='umm-score-test-') as td:
            det=Detector('.',td,'x');calls=[]
            def compute(*a):calls.append(1);return {'box':[],'score':[],'center':[],'label':[]}
            det.detect=compute;im=Image.new('RGB',(32,32))
            det(im,'cat');det(im,'cat');self.assertEqual(len(calls),1)
            readonly=Detector('.',td,'x',readonly=True);self.assertEqual(readonly(im,'cat')['box'],[])
            with self.assertRaises(ValueError):readonly(im,'dog')

if __name__=='__main__':unittest.main()

import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('sixrun_queue', Path(__file__).parents[1]/'scripts/sixrun_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_dependencies_and_single_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            queue.write(base/'plan.json', {'runs': {'mice_bare': {}}})
            jobs = [{'id': 'i', 'run': 'mice_bare', 'kind': 'infer', 'state': 'completed'}]
            queue.advance(base, jobs)
            queue.advance(base, jobs)
            self.assertEqual(sum(j['kind']=='audit' for j in jobs), 1)
            jobs[-1]['state']='completed'
            queue.advance(base, jobs)
            jobs[-1]['state']='completed'
            queue.advance(base, jobs)
            self.assertEqual(sum(j['kind']=='score' for j in jobs), 8)
            self.assertFalse(any(j['kind']=='aggregate' for j in jobs))
            for j in jobs:j['state']='completed'
            queue.advance(base,jobs)
            self.assertEqual(jobs[-1]['kind'],'aggregate')

    def test_preserves_uncommitted_transaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'turn_1.png').write_bytes(b'original interrupted evidence')
            queue.recover_transaction(folder)
            self.assertFalse((folder/'turn_1.png').exists())
            self.assertEqual(next(folder.glob('interrupted_transactions/*/turn_1.png')).read_bytes(), b'original interrupted evidence')

    def test_refuses_damaged_committed_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'turn_1.json').write_text('{}')
            with self.assertRaises(ValueError):queue.recover_transaction(folder)

    def test_reserves_inference_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            queue.write(base/'plan.json', {'runs': {}})
            jobs=[{'id': str(i),'kind':'score','run':'edival_bare','state':'running'} for i in range(2)]
            jobs += [{'id':'score','kind':'score','run':'edival_bare','state':'pending'},
                     {'id':'infer','kind':'infer','run':'edival_chat','state':'pending'}]
            queue.write(base/'queue.json',jobs)
            claimed=queue.claim(SimpleNamespace(base=base,host='a800_1',gpu='0'))
            self.assertEqual(claimed['id'],'infer')


if __name__=='__main__':unittest.main()

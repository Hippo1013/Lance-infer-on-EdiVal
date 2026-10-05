import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


script = Path(__file__).resolve().parents[1] / "scripts" / "performance_acceptance.py"
spec = importlib.util.spec_from_file_location("performance_acceptance", script)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class WorkerWaitTests(unittest.TestCase):
    def test_finished_worker_is_allowed_while_peer_saves_last_result(self):
        with tempfile.TemporaryDirectory(prefix="lance-worker-test-") as folder:
            left, right = (Path(folder) / name for name in ("left.json", "right.json"))
            left.write_text(json.dumps({"worker": 0}))

            class Finished:
                returncode = 0
                def poll(self):
                    return 0

            class Finishing:
                returncode = None
                def poll(self):
                    right.write_text(json.dumps({"worker": 1}))
                    return None

            self.assertEqual(acceptance.wait_files([left, right], [Finished(), Finishing()]),
                             [{"worker": 0}, {"worker": 1}])

    def test_exit_without_own_result_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lance-worker-test-") as folder:
            class MissingResult:
                returncode = 0
                def poll(self):
                    return 0

            with self.assertRaises(RuntimeError):
                acceptance.wait_files([Path(folder) / "missing.json"], [MissingResult()])

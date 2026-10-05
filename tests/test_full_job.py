import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('full_job', Path(__file__).resolve().parents[1]/'scripts/full_mice_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class FullJobTests(unittest.TestCase):
    def test_never_restore_on_occupied_gpu(self):
        with patch.object(job.subprocess, 'check_output', return_value='16000'), patch.object(job.subprocess, 'run') as run:
            result = job.restore_burn()
            run.assert_not_called()
            self.assertTrue(all('occupied' in x for x in result.values()))

    def test_restore_only_idle_0_1_panes_and_never_monitor(self):
        def output(command, **kw):
            return '0' if command[0] == 'nvidia-smi' else '1001'
        def run(command, **kw):
            return subprocess.CompletedProcess(command, 1 if command[0] == 'pgrep' else 0)
        with patch.object(job.subprocess, 'check_output', side_effect=output), patch.object(job.subprocess, 'run', side_effect=run) as call:
            job.restore_burn()
            sent = [x.args[0] for x in call.call_args_list if x.args[0][0] == 'tmux']
            self.assertEqual([x[3] for x in sent], ['0:0.0','1:0.0'])
            self.assertTrue(all('monitor' not in str(x) and 'watch_dog' not in str(x) for x in sent))

    def test_skip_busy_pane_even_if_gpu_empty(self):
        with patch.object(job.subprocess,'check_output',return_value='0'), patch.object(job.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run:
            job.restore_burn()
            self.assertTrue(all(x.args[0][0]=='pgrep' for x in run.call_args_list))

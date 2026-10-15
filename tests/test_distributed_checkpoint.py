import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DistributedCheckpointTest(unittest.TestCase):
    def test_all_ranks_select_published_checkpoint_when_writer_is_delayed(self):
        project = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'checkpoint-output'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       PYTHONPATH=str(project / 'src'))
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=2', str(project / 'tests/checkpoint_fixture.py'), str(output)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            selected = [json.loads((output / f'rank-{rank}.json').read_text())['best'] for rank in range(2)]
            self.assertEqual(selected, [str(output / 'checkpoint-1')] * 2)


if __name__ == '__main__':
    unittest.main()

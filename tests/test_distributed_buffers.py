import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DistributedBuffersTest(unittest.TestCase):
    def test_ddp_preserves_modalities_with_different_python_hash_seeds(self):
        project = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            store = (Path(directory) / 'ddp-store').as_uri()
            processes = []
            for rank in range(2):
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                    PYTHONPATH=str(project / 'src'), PYTHONHASHSEED=str(rank),
                    RANK=str(rank), DDP_STORE=store,
                )
                processes.append(subprocess.Popen(
                    [sys.executable, str(project / 'tests/buffer_fixture.py')],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                ))
            results = []
            try:
                for process in processes:
                    stdout, stderr = process.communicate(timeout=60)
                    results.append((process.returncode, stdout, stderr))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
            for rank, (code, stdout, stderr) in enumerate(results):
                self.assertEqual(code, 0, f'rank {rank}: {stdout}\n{stderr}')


if __name__ == '__main__':
    unittest.main()

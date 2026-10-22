"""Job-local temporary directories retain a common GPU reservation scope."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherRuntimeTest(unittest.TestCase):
    def make_launcher(self, directory):
        project = Path(directory)
        scripts = project / 'scripts'
        scripts.mkdir()
        source = Path(__file__).parents[1] / 'scripts/run_lambda_queue.sh'
        header = source.read_text().split('datasets=("$@")')[0]
        launcher = scripts / 'runtime.sh'
        launcher.write_text(header + '\nprintf "%s\\n" "$TMPDIR"\n')
        return launcher

    def test_jobs_isolate_temporary_files_from_shared_tmpdir(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = self.make_launcher(directory)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TMPDIR='/deleted/shared/tmp')
            paths = [subprocess.check_output(['bash', str(launcher)], env=env, text=True).strip()
                     for _ in range(2)]
            self.assertNotEqual(paths[0], paths[1])
            for value in paths:
                path = Path(value)
                self.assertEqual(path.parent, Path(directory) / '.runtime')
                self.assertTrue(path.is_dir())
                self.assertLess(len(str(path / 'pymp-12345678/listener-12345678')), 108)

    def test_gpu_lock_is_shared_across_private_runtime_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = self.make_launcher(directory)
            launcher.write_text(launcher.read_text() + 'read -r release\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
            first = subprocess.Popen(['bash', str(launcher)], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(first.stdout.readline().strip())
                second = subprocess.run(['bash', str(launcher)], env=env,
                                        capture_output=True, text=True, timeout=5)
                self.assertNotEqual(second.returncode, 0)
                self.assertIn('reserved GPU 0', second.stdout)
            finally:
                first.communicate('release\n', timeout=5)


if __name__ == '__main__':
    unittest.main()

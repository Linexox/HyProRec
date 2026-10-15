import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('refresh_queue', Path(__file__).parents[1] / 'scripts/run_refresh_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class RefreshQueueTest(unittest.TestCase):
    def test_successful_stage_is_not_run_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, '-c', "from pathlib import Path; p=Path('count'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"]
            queue.run_stage('first', command, root, root / 'state.json', .01)
            queue.run_stage('first', command, root, root / 'state.json', .01)
            self.assertEqual((root / 'count').read_text(), 'x')
            self.assertEqual(json.loads((root / 'state.json').read_text())['stages']['first']['exit_code'], 0)

    def test_failure_prevents_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                queue.run_stage('first', [sys.executable, '-c', 'raise SystemExit(7)'], root, root / 'state.json', .01)
                queue.run_stage('second', [sys.executable, '-c', "open('unexpected','w').close()"], root, root / 'state.json', .01)
            self.assertFalse((root / 'unexpected').exists())
            self.assertEqual(json.loads((root / 'state.json').read_text())['stages']['first']['exit_code'], 7)

    def test_restart_skips_completed_trial_and_resumes_failed_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state.json'
            first = [sys.executable, '-c', "from pathlib import Path; p=Path('count'); p.write_text(p.read_text()+'1' if p.exists() else '1')"]
            second = [sys.executable, '-c', "from pathlib import Path; import sys; sys.exit(7) if Path('fail').exists() else None; p=Path('count'); p.write_text(p.read_text()+'2')"]
            (root / 'fail').touch()
            queue.run_stage('lambda-0p1', first, root, state, .01)
            with self.assertRaises(subprocess.CalledProcessError):
                queue.run_stage('lambda-0p3', second, root, state, .01)
            (root / 'fail').unlink()
            queue.run_stage('lambda-0p1', first, root, state, .01)
            queue.run_stage('lambda-0p3', second, root, state, .01, retry_failed=True)
            self.assertEqual((root / 'count').read_text(), '12')

    def test_config_update_preserves_editable_comments_and_changes_only_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directory = root / 'configs/redial'
            directory.mkdir(parents=True)
            config = directory / 'full-lambda-0p1.yaml'
            config.write_text('model:\n  grounding_checkpoint_path: old\n  conversation_loss_weight: 0.1 # keep\ndata:\n  dataset_path: data/hyprorec-redial\n  embeddings_dir_name: embeddings\n  hyperedge_table_path: old-table\ntraining:\n  output_dir: outputs/redial/full-lambda-0p1\n')
            queue.update_configs(root, 'version-1', {'redial':'new-grounding'})
            text = config.read_text()
            self.assertIn('conversation_loss_weight: 0.1 # keep', text)
            self.assertIn('grounding_checkpoint_path: new-grounding', text)
            self.assertIn('embeddings_dir_name: feature_versions/version-1/embeddings', text)
            self.assertIn('hyperedge_table_path: data/hyprorec-redial/feature_versions/version-1/hyperedge_table.json', text)

    def test_cleanup_rejects_escape_before_deleting_any_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / 'outputs'
            outputs.mkdir()
            safe = outputs / 'old-checkpoint'
            safe.mkdir()
            unsafe = root / 'backbone'
            unsafe.write_text('keep')
            manifest = {'root':str(outputs), 'targets':[{'path':str(safe),'kind':'directory'}, {'path':str(unsafe),'kind':'file'}]}
            with self.assertRaises(ValueError):
                queue.clean_retired(manifest, outputs)
            self.assertTrue(safe.exists())
            self.assertEqual(unsafe.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()

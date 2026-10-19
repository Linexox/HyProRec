import os
import tempfile
import unittest
from unittest.mock import patch

from accelerate import ParallelismConfig
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from hyprorec.arguments import HoCRSTrainingArguments


class DistributedArgumentsTest(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def build(self, replicate: int, shard: int) -> HoCRSTrainingArguments:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        return HoCRSTrainingArguments(
            output_dir=self.directory.name,
            use_cpu=True,
            dp_replicate_size=replicate,
            dp_shard_size=shard,
        )

    def test_ddp_uses_no_custom_mesh(self) -> None:
        arguments = self.build(1, 1)
        self.assertIsNone(arguments.parallelism_config)
        self.assertEqual(arguments.fsdp, [])

    def test_fsdp2_mesh(self) -> None:
        arguments = self.build(1, 2)
        self.assertEqual(arguments.fsdp_config["version"], 2)
        self.assertEqual(arguments.parallelism_config.dp_replicate_size, 1)
        self.assertEqual(arguments.parallelism_config.dp_shard_size, 2)

    def test_hsdp_mesh(self) -> None:
        arguments = self.build(2, 2)
        self.assertEqual(arguments.parallelism_config.dp_replicate_size, 2)
        self.assertEqual(arguments.parallelism_config.dp_shard_size, 2)
        self.assertEqual(arguments.parallelism_config.total_size, 4)

    def test_fsdp_settings_reach_accelerate_plugin_and_mesh(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.build(1, 2)
            plugin = FullyShardedDataParallelPlugin(sync_module_states=False)
            self.assertEqual(plugin.fsdp_version, 2)
            self.assertIs(plugin.auto_wrap_policy, transformer_auto_wrap_policy)
            self.assertEqual(ParallelismConfig().dp_shard_size, 2)
            self.assertEqual(os.environ['ACCELERATE_USE_PARALLELISM_CONFIG'], 'true')
            self.assertTrue(plugin.reshard_after_forward)

    def test_replication_only_requires_plain_ddp(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordinary DDP"):
            self.build(2, 1)


if __name__ == "__main__":
    unittest.main()

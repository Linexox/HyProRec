import tempfile
import unittest

from hyprorec.arguments import HoCRSTrainingArguments


class DistributedArgumentsTest(unittest.TestCase):
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
        self.assertIsNone(arguments.fsdp)

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

    def test_replication_only_requires_plain_ddp(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordinary DDP"):
            self.build(2, 1)


if __name__ == "__main__":
    unittest.main()

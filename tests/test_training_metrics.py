import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import TrainingArguments

from hyprorec.training import HyProRecTrainer


class ScalarLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels=None, rec_labels=None):
        value = input_ids.float().mean() + self.offset
        return SimpleNamespace(
            loss=10.4 * value,
            fusion_loss=value,
            view_loss=2 * value,
            conversation_loss=3 * value,
            recommendation_logits=torch.zeros(input_ids.size(0), 8, device=input_ids.device),
            conversation_predictions=torch.zeros_like(input_ids),
            route_weights=None,
            attention_weights=None,
        )


class MetricRecorder:
    def __init__(self):
        self.records = []

    def log_trainer(self, stage, logs):
        self.records.append((stage, logs))


def make_trainer(output):
    dataset = [dict(input_ids=torch.tensor([value]), labels=torch.tensor([0]),
                    rec_labels=torch.tensor(0)) for value in range(1, 6)]
    recorder = MetricRecorder()
    trainer = HyProRecTrainer(
        model=ScalarLossModel(),
        args=TrainingArguments(
            output_dir=str(output), use_cpu=True,
            ddp_backend="gloo" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
            per_device_eval_batch_size=1, report_to=[], disable_tqdm=True,
        ),
        eval_dataset=dataset,
        swanlab_tracker=recorder,
    )
    return trainer, recorder, dataset


class TrainingMetricsTest(unittest.TestCase):
    def test_fsdp2_autocast_handles_float_features_and_bf16_weights(self):
        with tempfile.TemporaryDirectory() as output:
            trainer, _, _ = make_trainer(output)
            trainer.is_fsdp_enabled = True
            trainer.args.bf16 = True
            trainer.use_cpu_amp = False
            trainer.accelerator.state.fsdp_plugin = SimpleNamespace(fsdp_version=2)
            layer = torch.nn.Linear(2, 2).to(torch.bfloat16)
            try:
                with trainer.compute_loss_context_manager():
                    result = layer(torch.ones(1, 2))
                self.assertEqual(result.dtype, torch.bfloat16)
            finally:
                trainer.is_fsdp_enabled = False
                del trainer.accelerator.state.fsdp_plugin

    def setUp(self):
        environment = patch.dict(os.environ, {
            "ACCELERATE_USE_FSDP": "false", "ACCELERATE_USE_DEEPSPEED": "false",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_validation_excludes_unlogged_training_and_preserves_training_window(self):
        with tempfile.TemporaryDirectory() as output:
            trainer, recorder, _ = make_trainer(output)
            trainer.model.train()
            trainer.compute_loss(trainer.model, dict(input_ids=torch.tensor([[-100]])))
            trainer.evaluate()
            stage, metrics = recorder.records[-1]
            self.assertEqual(stage, "eval")
            self.assertAlmostEqual(metrics["loss/fusion"], 3.0, places=5)
            trainer.log({"loss": -1040.0})
            self.assertEqual(recorder.records[-1][0], "train")
            self.assertAlmostEqual(recorder.records[-1][1]["loss/fusion"], -100.0)

    def test_repeated_validation_and_test_return_consistent_loss_components(self):
        with tempfile.TemporaryDirectory() as output:
            trainer, recorder, dataset = make_trainer(output)
            for metrics, prefix in [
                (trainer.evaluate(), "eval"),
                (trainer.evaluate(), "eval"),
                (trainer.predict(dataset).metrics, "test"),
            ]:
                self.assertAlmostEqual(metrics[f"{prefix}_loss/fusion"], 3.0, places=5)
                self.assertAlmostEqual(metrics[f"{prefix}_loss/view"], 6.0, places=5)
                self.assertAlmostEqual(metrics[f"{prefix}_loss/conversation"], 9.0, places=5)
                self.assertEqual(metrics[f"{prefix}_loss/total"], metrics[f"{prefix}_loss"])
                self.assertAlmostEqual(metrics[f"{prefix}_loss"], 31.2, places=4)
            self.assertEqual([stage for stage, _ in recorder.records], ["eval", "eval", "test"])

    def test_diagnostic_collection_preserves_loss_gradient(self):
        with tempfile.TemporaryDirectory() as output:
            trainer, _, _ = make_trainer(output)
            inputs = dict(input_ids=torch.tensor([[2]]))
            trainer.compute_loss(trainer.model, inputs).backward()
            self.assertAlmostEqual(trainer.model.offset.grad.item(), 10.4, places=5)

    def test_four_ranks_exclude_padding_and_report_same_global_losses(self):
        project = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone",
                 "--nproc-per-node=4", str(project / "tests/metrics_fixture.py"), output],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                         ACCELERATE_USE_FSDP="false", ACCELERATE_USE_DEEPSPEED="false",
                         PYTHONPATH=str(project / "src")),
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            for rank in range(4):
                metrics = json.loads((Path(output) / f"rank-{rank}.json").read_text())
                self.assertAlmostEqual(metrics["eval_loss/fusion"], 3.0, places=5)
                self.assertAlmostEqual(metrics["eval_loss/view"], 6.0, places=5)
                self.assertAlmostEqual(metrics["eval_loss/conversation"], 9.0, places=5)
                self.assertEqual(metrics["eval_loss/total"], metrics["eval_loss"])
                self.assertAlmostEqual(metrics["eval_loss"], 31.2, places=4)
                if rank == 0:
                    self.assertAlmostEqual(metrics["train_fusion"], 2.5, places=5)


if __name__ == "__main__":
    unittest.main()

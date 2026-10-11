import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from hyprorec.data.redial import HoCRSDataCollator
from hyprorec.scripts.grounding import GroundingTrainer, _finalize_grounding
from hyprorec.tracking import SwanLabTracker
from hyprorec.training import (
    HyProRecTrainer,
    StageAwareTrainerMixin,
    canonical_stage_logs,
)


class _FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if isinstance(text, list):
            values = [[len(item)] for item in text]
        else:
            values = [len(text)]
        return {"input_ids": values}

    def decode(self, values, **kwargs):
        value = values[0] if values else 0
        return "x" * (value if isinstance(value, int) else value[0])

    def pad(self, prompts, **kwargs):
        width = max(len(item["input_ids"]) for item in prompts)
        input_ids = [
            item["input_ids"] + [0] * (width - len(item["input_ids"]))
            for item in prompts
        ]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(
                [
                    [1] * len(item["input_ids"])
                    + [0] * (width - len(item["input_ids"]))
                    for item in prompts
                ]
            ),
        }


class _FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_prompt(self, context, graph_sizes):
        return f"{context}|{graph_sizes}"

    def get_token_id_map(self):
        return {
            "soft_prompt_token_id": 3,
            "graph_start_token_ids": {},
            "graph_end_token_ids": {},
            "node_token_id": 4,
            "hyperedge_token_id": 5,
        }


class CollatorCacheTest(unittest.TestCase):
    def test_repeated_sample_reuses_prompt_and_response_tokens(self):
        tokenizer = _FakeTokenizer()
        collator = HoCRSDataCollator(_FakeProcessor(tokenizer))
        sample = {
            "context": "history",
            "response": "response",
            "role": "assistant",
            "target_item_id": 0,
            "hypergraphs": {},
        }

        collator([sample])
        collator([sample])

        self.assertEqual(len(collator._prompt_cache), 1)
        self.assertEqual(len(collator._response_cache), 1)

    def test_token_caches_are_bounded(self):
        tokenizer = _FakeTokenizer()
        collator = HoCRSDataCollator(_FakeProcessor(tokenizer), cache_size=2)
        for index in range(3):
            sample = {
                "context": f"history-{index}{'x' * index}",
                "response": f"response-{index}",
                "role": "assistant",
                "target_item_id": 0,
                "hypergraphs": {},
            }
            collator([sample])

        self.assertLessEqual(len(collator._prompt_cache), 2)
        self.assertLessEqual(len(collator._response_cache), 2)
        self.assertNotIn(("x" * 8, (), "assistant"), collator._prompt_cache)


class SwanLabProcessTest(unittest.TestCase):
    def test_nonzero_rank_does_not_create_a_run(self):
        @dataclass
        class Model:
            backbone_name_or_path: str = "model"
            recommendation_hidden_dim: int = 1
            num_prompt_tokens: int = 20
            conversation_loss_weight: float = 1.0
            recommendation_view_loss_weight: float = 0.2

        @dataclass
        class Data:
            dataset_path: str = "data"
            views: list[str] = None
            max_hypergraph_nodes: int = 64

            def __post_init__(self):
                self.views = ["txt"]

        @dataclass
        class Training:
            run_name: str = "only-txt"
            seed: int = 42

            def to_dict(self):
                return {}

        with patch.dict(os.environ, {"RANK": "1", "SWANLAB_MODE": "online"}, clear=False):
            with patch("swanlab.init") as init:
                tracker = SwanLabTracker.start(
                    Model(),
                    Data(),
                    Training(),
                    None,
                )

        self.assertIsNone(tracker.run)
        init.assert_not_called()


class SwanLabStageTest(unittest.TestCase):
    def test_tracker_receives_explicit_eval_stage_without_prefix_adaptation(self):
        tracker = SwanLabTracker(run=object())
        recorded = []
        tracker.log = lambda metrics: recorded.append(metrics)

        tracker.log_trainer(
            "eval",
            {
                "loss/total": 1.0,
                "attention/txt/context_mass": 0.25,
                "rec_router/mean_txt": 0.5,
            },
        )
        tracker.log_grounding(
            "eval",
            {
                "loss": 2.0,
                "ga_sa_loss": 0.75,
            },
            view="txt",
        )

        self.assertEqual(
            recorded[0],
            {
                "eval/loss/total": 1.0,
                "other_eval/AttnPooling/txt-context-mass": 0.25,
                "other_eval/RecRouter/txt": 0.5,
            },
        )
        self.assertEqual(
            recorded[1],
            {
                "eval/grounding/txt/total": 2.0,
                "eval/grounding/txt/ga-sa": 0.75,
            },
        )

    def test_grounding_metrics_are_namespaced_by_view(self):
        tracker = SwanLabTracker(run=object())
        recorded = []
        tracker.log = lambda metrics: recorded.append(metrics)

        tracker.log_grounding(
            "train",
            {"loss": 1.0, "ga_sn_loss": 0.5},
            view="img",
        )

        self.assertEqual(
            recorded,
            [{
                "train/grounding/img/total": 1.0,
                "train/grounding/img/ga-sn": 0.5,
            }],
        )

    def test_trainer_stage_is_explicit_at_evaluation_boundary(self):
        class BaseTrainer:
            def __init__(self):
                self.stages = []

            def evaluate(self):
                self.stages.append(self._metric_stage)
                return {}

        class TestTrainer(StageAwareTrainerMixin, BaseTrainer):
            pass

        trainer = TestTrainer()
        trainer.evaluate()

        self.assertEqual(trainer.stages, ["eval"])
        self.assertEqual(trainer._metric_stage, "train")
        self.assertEqual(
            canonical_stage_logs("eval", {"eval_loss": 1.0, "loss/total": 2.0}),
            {"loss": 1.0, "loss/total": 2.0},
        )


class GroundingProcessTest(unittest.TestCase):
    def test_grounding_loss_accumulators_are_isolated_by_stage(self):
        trainer = GroundingTrainer.__new__(GroundingTrainer)
        trainer._metric_stage = "train"
        trainer._grounding_sum = {}
        trainer._grounding_count = {}
        train_outputs = type(
            "Outputs",
            (),
            {
                "loss": torch.tensor(100.0),
                "ga_sa_loss": torch.tensor(10.0),
                "ga_sn_loss": None,
                "sa_sn_loss": None,
            },
        )()
        trainer._accumulate_grounding_losses(train_outputs)

        trainer._metric_stage = "eval"
        eval_outputs = type(
            "Outputs",
            (),
            {
                "loss": torch.tensor(2.0),
                "ga_sa_loss": torch.tensor(1.0),
                "ga_sn_loss": None,
                "sa_sn_loss": None,
            },
        )()
        trainer._accumulate_grounding_losses(eval_outputs)

        self.assertEqual(trainer._grounding_count, {"train": 1, "eval": 1})
        self.assertEqual(trainer._grounding_sum["train"]["loss"], 100.0)
        self.assertEqual(trainer._grounding_sum["eval"]["loss"], 2.0)

    def test_nonzero_rank_skips_combined_checkpoint_finalization(self):
        self.assertIsNone(
            _finalize_grounding(
                False,
                None,
                None,
                {},
                {},
                None,
                None,
                None,
                None,
            )
        )


class DiagnosticCostTest(unittest.TestCase):
    def test_attention_diagnostics_only_collect_required_masses(self):
        trainer = HyProRecTrainer.__new__(HyProRecTrainer)
        trainer.attention_token_ids = {
            "soft_prompt_token_id": 3,
            "graph_start_token_ids": {"txt": 6},
            "graph_end_token_ids": {"txt": 7},
            "node_token_id": 4,
            "hyperedge_token_id": 5,
        }
        recorded = []
        trainer._record = lambda key, value, prefix="": recorded.append(key)
        trainer._attention_diagnostics(
            {
                "input_ids": torch.tensor([[3, 6, 4, 5, 7, 8]]),
                "pooling_mask": torch.ones(1, 6, dtype=torch.bool),
            },
            {"txt": torch.full((1, 6), 1 / 6)},
        )

        self.assertIn("attention/txt/soft_prompt_mass", recorded)
        self.assertIn("attention/txt/context_mass", recorded)
        self.assertIn("attention/txt/hypergraph_mass", recorded)
        self.assertFalse(
            any(
                key.endswith(("/entropy", "/top5_mass", "/max_weight"))
                for key in recorded
            )
        )


if __name__ == "__main__":
    unittest.main()

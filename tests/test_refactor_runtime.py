import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from hyprorec.data.redial import HoCRSDataCollator
from hyprorec.scripts.grounding import _finalize_grounding
from hyprorec.tracking import SwanLabTracker


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
        input_ids = [item["input_ids"] + [0] * (width - len(item["input_ids"])) for item in prompts]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(
                [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in prompts]
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


class GroundingProcessTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

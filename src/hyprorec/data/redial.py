"""ReDial samples and a graph-aware Transformers data collator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ..constants import GRAPH_VIEWS
from ..processing_hocrs import HoCRSProcessor
from .batch import BatchData, HypergraphBatch, batch_hypergraphs
from .hypergraph import HypergraphData, HypergraphTable


@dataclass
class HoCRSDatasetConfig:
    dataset_path: str
    hyperedge_table_path: str | None = None
    views: tuple[str, ...] = GRAPH_VIEWS
    topk: int = 3
    khop: int = 2

    def __post_init__(self) -> None:
        self.views = tuple(dict.fromkeys(self.views))
        unknown_views = set(self.views) - set(GRAPH_VIEWS)
        if unknown_views:
            raise ValueError(f"Unknown graph views: {sorted(unknown_views)}")


class HoCRSDataset(Dataset):
    """Turn-level recommendation examples with context-conditioned local graphs."""

    SPLIT_FILES = {
        "train": "train_data.json",
        "validation": "valid_data.json",
        "test": "test_data.json",
    }

    def __init__(
        self,
        config: HoCRSDatasetConfig,
        split: str,
        hypergraph_table: HypergraphTable | None = None,
    ) -> None:
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unknown split: {split}")
        self.config = config
        self.split = split
        dataset_path = Path(config.dataset_path)
        table_path = (
            config.hyperedge_table_path or dataset_path / "hyperedge_table.json"
        )
        # START: Allow no-hypergraph ablations without requiring a topology file.
        self.hypergraph_table = (
            hypergraph_table or HypergraphTable.from_json(table_path)
            if config.views
            else None
        )
        # END: Allow no-hypergraph ablations without requiring a topology file.
        with (dataset_path / self.SPLIT_FILES[split]).open(encoding="utf-8") as file:
            conversations = json.load(file)
        self.samples = self._build_samples(conversations)

    @staticmethod
    def _build_samples(conversations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for conversation in conversations:
            context: list[str] = []
            context_item_ids: list[int] = []
            for turn in conversation["dialog"]:
                turn_items = [int(item_id) for item_id in turn.get("items", [])]
                if turn["role"] == "Recommender" and context_item_ids and turn_items:
                    for target_item_id in turn_items:
                        samples.append(
                            {
                                "context": "\n".join(context),
                                "context_item_ids": list(
                                    dict.fromkeys(context_item_ids)
                                ),
                                "target_item_id": target_item_id,
                                "response": turn["text"],
                            }
                        )
                context.append(f"[{turn['role']}]: {turn['text']}")
                context_item_ids.extend(turn_items)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if not self.config.views:
            return {**sample, "hypergraphs": {}}
        assert self.hypergraph_table is not None
        graphs = {
            view: self.hypergraph_table.build_local(
                sample["context_item_ids"],
                view=view,
                topk=self.config.topk,
                khop=self.config.khop,
            )
            for view in self.config.views
        }
        return {**sample, "hypergraphs": graphs}


class HoCRSDataCollator:
    """Tokenize conversations and pack every graph view into tensor mappings."""

    def __init__(
        self, processor: HoCRSProcessor, max_length: int | None = None
    ) -> None:
        self.processor = processor
        self.max_length = max_length

    @staticmethod
    def _positions(
        input_ids: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        value_token_id: int,
        expected_counts: Sequence[int],
    ) -> torch.Tensor:
        positions: list[tuple[int, int]] = []
        for row_index, expected_count in enumerate(expected_counts):
            row = input_ids[row_index]
            starts = torch.nonzero(row == start_token_id, as_tuple=False).flatten()
            ends = torch.nonzero(row == end_token_id, as_tuple=False).flatten()
            if starts.numel() != 1 or ends.numel() != 1 or starts[0] >= ends[0]:
                raise ValueError(
                    "Each graph view must have one complete serialized block."
                )
            columns = torch.nonzero(
                (row == value_token_id)
                & (torch.arange(row.numel()) > starts[0])
                & (torch.arange(row.numel()) < ends[0]),
                as_tuple=False,
            ).flatten()
            if columns.numel() != expected_count:
                raise ValueError(
                    "Graph placeholders were truncated or do not match the topology: "
                    f"expected {expected_count}, found {columns.numel()}."
                )
            positions.extend((row_index, int(column)) for column in columns)
        return torch.tensor(positions, dtype=torch.long)

    def __call__(self, features: Sequence[dict[str, Any]]) -> BatchData:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        views = tuple(features[0]["hypergraphs"])
        if any(tuple(feature["hypergraphs"]) != views for feature in features):
            raise ValueError("All samples in a batch must enable the same graph views.")

        hypergraphs: dict[str, HypergraphBatch] = {}
        graph_sizes: list[dict[str, tuple[int, int]]] = []
        for feature in features:
            graph_sizes.append(
                {
                    view: (
                        feature["hypergraphs"][view].num_nodes,
                        feature["hypergraphs"][view].num_hyperedges,
                    )
                    for view in views
                }
            )
        prompts = [
            self.processor.build_prompt(feature["context"], sizes)
            for feature, sizes in zip(features, graph_sizes)
        ]
        eos_token = self.processor.tokenizer.eos_token or ""
        texts = [
            f"{prompt}\n{feature['response']}{eos_token}"
            for prompt, feature in zip(prompts, features)
        ]
        tokenize_kwargs = {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            tokenize_kwargs["max_length"] = self.max_length
        encoded = self.processor(texts, **tokenize_kwargs)
        full_prompt_lengths = [
            len(
                self.processor.tokenizer(
                    f"{prompt}\n",
                    add_special_tokens=False,
                )["input_ids"]
            )
            for prompt in prompts
        ]
        full_text_lengths = [
            len(
                self.processor.tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]
            )
            for text in texts
        ]

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row_index, (prompt_length, full_length) in enumerate(
            zip(full_prompt_lengths, full_text_lengths)
        ):
            retained_length = int(encoded["attention_mask"][row_index].sum())
            truncated_prefix = max(0, full_length - retained_length)
            retained_prompt_length = max(0, prompt_length - truncated_prefix)
            labels[row_index, :retained_prompt_length] = -100

        token_ids = self.processor.get_token_id_map()
        for view in views:
            graphs: list[HypergraphData] = [
                feature["hypergraphs"][view] for feature in features
            ]
            graph_batch = batch_hypergraphs(graphs)
            graph_batch["node_positions"] = self._positions(
                encoded["input_ids"],
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["node_token_id"],
                [graph.num_nodes for graph in graphs],
            )
            graph_batch["hyperedge_positions"] = self._positions(
                encoded["input_ids"],
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["hyperedge_token_id"],
                [graph.num_hyperedges for graph in graphs],
            )
            hypergraphs[view] = graph_batch

        return BatchData(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
                "rec_labels": torch.tensor(
                    [feature["target_item_id"] for feature in features],
                    dtype=torch.long,
                ),
                "hypergraphs": hypergraphs,
            }
        )


__all__ = ["HoCRSDataCollator", "HoCRSDataset", "HoCRSDatasetConfig"]

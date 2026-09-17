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
        table_path = dataset_path / "hyperedge_table.json"
        self.hypergraph_table = (
            hypergraph_table or HypergraphTable.from_json(table_path)
            if config.views
            else None
        )
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
                if turn["role"] == "Recommender" and turn_items:
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
        if not self.config.views or not sample["context_item_ids"]:
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

    def __init__(self, processor: HoCRSProcessor) -> None:
        self.processor = processor

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
            if expected_count == 0 and starts.numel() == 0 and ends.numel() == 0:
                continue
            if starts.numel() != 1 or ends.numel() != 1 or starts[0] >= ends[0]:
                raise ValueError("Each graph view must have one complete serialized block.")
            columns = torch.nonzero(
                (row == value_token_id)
                & (torch.arange(row.numel()) > starts[0])
                & (torch.arange(row.numel()) < ends[0]),
                as_tuple=False,
            ).flatten()
            if columns.numel() != expected_count:
                raise ValueError(
                    "Graph placeholders do not match the topology: "
                    f"expected {expected_count}, found {columns.numel()}."
                )
            positions.extend((row_index, int(column)) for column in columns)
        return torch.tensor(positions, dtype=torch.long).reshape(-1, 2)

    def _prepare_sample(
        self,
        feature: dict[str, Any],
    ) -> tuple[dict[str, HypergraphData], str]:
        graphs = dict(feature["hypergraphs"])
        sizes = {
            view: (graph.num_nodes, graph.num_hyperedges)
            for view, graph in graphs.items()
        }
        return graphs, self.processor.build_prompt(feature["context"], sizes)

    @staticmethod
    def _collect_views(
        graph_sets: Sequence[dict[str, HypergraphData]],
    ) -> list[str]:
        views: list[str] = []
        for graphs in graph_sets:
            for view in graphs:
                if view not in views:
                    views.append(view)
        return views

    def _encode_batch(
        self,
        features: Sequence[dict[str, Any]],
        prompts: Sequence[str],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        eos_token = self.processor.tokenizer.eos_token or ""
        batch_input_ids: list[list[int]] = []
        prompt_lengths: list[int] = []

        for feature, prompt in zip(features, prompts):
            prompt_text = f"{prompt}\n"
            text = f"{prompt_text}{feature['response']}{eos_token}"
            prompt_length = len(
                self.processor.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            input_ids = self.processor.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            batch_input_ids.append(input_ids)
            prompt_lengths.append(prompt_length)

        encoded = self.processor.tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in batch_input_ids],
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100
        return encoded, labels

    def _batch_graph_views(
        self,
        graph_sets: Sequence[dict[str, HypergraphData]],
        views: Sequence[str],
        input_ids: torch.Tensor,
        token_ids: dict[str, Any],
    ) -> dict[str, HypergraphBatch]:
        hypergraphs: dict[str, HypergraphBatch] = {}
        for view in views:
            graphs_by_row = [graph_set.get(view) for graph_set in graph_sets]
            graphs = [graph for graph in graphs_by_row if graph is not None]
            graph_batch = batch_hypergraphs(graphs)
            graph_batch["node_positions"] = self._positions(
                input_ids,
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["node_token_id"],
                [graph.num_nodes if graph is not None else 0 for graph in graphs_by_row],
            )
            graph_batch["hyperedge_positions"] = self._positions(
                input_ids,
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["hyperedge_token_id"],
                [
                    graph.num_hyperedges if graph is not None else 0
                    for graph in graphs_by_row
                ],
            )
            hypergraphs[view] = graph_batch
        return hypergraphs

    def __call__(self, features: Sequence[dict[str, Any]]) -> BatchData:
        if not features:
            raise ValueError("Cannot collate an empty batch.")

        prepared = [self._prepare_sample(feature) for feature in features]
        graph_sets = [graphs for graphs, _ in prepared]
        prompts = [prompt for _, prompt in prepared]
        views = self._collect_views(graph_sets)
        token_ids = self.processor.get_token_id_map()
        encoded, labels = self._encode_batch(features, prompts)
        hypergraphs = self._batch_graph_views(
            graph_sets,
            views,
            encoded["input_ids"],
            token_ids,
        )

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

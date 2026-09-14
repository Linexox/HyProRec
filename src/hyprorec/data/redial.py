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
        self,
        processor: HoCRSProcessor,
        max_length: int | None = 1024,
        max_history_tokens: int = 150,
        max_response_tokens: int = 64,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        # START: Reserve explicit budgets for history and response supervision.
        self.max_history_tokens = max_history_tokens
        self.max_response_tokens = max_response_tokens
        # END: Reserve explicit budgets for history and response supervision.

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

    def _truncate(
        self,
        input_ids: list[int],
        prompt_length: int,
        protected_token_id: int,
    ) -> tuple[list[int], int]:
        """Keep graph prompts intact while trimming context, then response."""

        response = input_ids[prompt_length:]
        # START: Apply the response token budget before the total sequence budget.
        response = response[: self.max_response_tokens]
        input_ids = input_ids[:prompt_length] + response
        # END: Apply the response token budget before the total sequence budget.
        if self.max_length is None or len(input_ids) <= self.max_length:
            return input_ids, prompt_length

        try:
            protected_start = input_ids.index(protected_token_id)
        except ValueError as error:
            raise ValueError(
                "The protected prompt boundary token is missing."
            ) from error

        prompt_prefix = input_ids[:protected_start]
        protected_prompt = input_ids[protected_start:prompt_length]
        response = input_ids[prompt_length:]
        response_budget = self.max_length - len(protected_prompt)
        if response_budget <= 0:
            raise ValueError(
                "The serialized graph prompt does not fit within max_length."
            )

        retained_response = response[:response_budget]
        prefix_budget = response_budget - len(retained_response)
        retained_prefix = prompt_prefix[-prefix_budget:] if prefix_budget else []
        retained_prompt_length = len(retained_prefix) + len(protected_prompt)
        return (
            retained_prefix + protected_prompt + retained_response,
            retained_prompt_length,
        )

    # START: Fit graph prompts by removing only complete BFS-tail hyperedges.
    def _fit_graphs(
        self,
        context: str,
        graphs: dict[str, HypergraphData],
    ) -> tuple[dict[str, HypergraphData], str]:
        while True:
            sizes = {
                view: (graph.num_nodes, graph.num_hyperedges)
                for view, graph in graphs.items()
            }
            prompt = self.processor.build_prompt(context, sizes)
            graph_prompt_length = len(
                self.processor.tokenizer(
                    f"{self.processor.build_prompt('', sizes)}\n",
                    add_special_tokens=False,
                )["input_ids"]
            )
            if self.max_length is None or graph_prompt_length <= self.max_length:
                return graphs, prompt

            candidates = [
                view for view, graph in graphs.items() if graph.num_hyperedges > 1
            ]
            if not candidates:
                return graphs, prompt
            view = max(
                candidates,
                key=lambda name: graphs[name].num_nodes + graphs[name].num_hyperedges,
            )
            graph = graphs[view]
            graphs[view] = graph.truncate_hyperedges(graph.num_hyperedges - 1)
    # END: Fit graph prompts by removing only complete BFS-tail hyperedges.

    def __call__(self, features: Sequence[dict[str, Any]]) -> BatchData:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        views = tuple(features[0]["hypergraphs"])
        if any(tuple(feature["hypergraphs"]) != views for feature in features):
            raise ValueError("All samples in a batch must enable the same graph views.")

        hypergraphs: dict[str, HypergraphBatch] = {}
        # START: Bound recent history and fit graph blocks before tokenization.
        fitted_graphs: list[dict[str, HypergraphData]] = []
        prompts: list[str] = []
        for feature in features:
            history_ids = self.processor.tokenizer(
                feature["context"],
                add_special_tokens=False,
            )["input_ids"]
            history_ids = history_ids[-self.max_history_tokens :]
            context = self.processor.tokenizer.decode(
                history_ids,
                skip_special_tokens=False,
            )
            graphs, prompt = self._fit_graphs(
                context,
                dict(feature["hypergraphs"]),
            )
            fitted_graphs.append(graphs)
            prompts.append(prompt)
        # END: Bound recent history and fit graph blocks before tokenization.
        eos_token = self.processor.tokenizer.eos_token or ""
        texts = [
            f"{prompt}\n{feature['response']}{eos_token}"
            for prompt, feature in zip(prompts, features)
        ]
        full_prompt_lengths = [
            len(
                self.processor.tokenizer(
                    f"{prompt}\n",
                    add_special_tokens=False,
                )["input_ids"]
            )
            for prompt in prompts
        ]
        tokenized_texts = [
            self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in texts
        ]
        token_ids = self.processor.get_token_id_map()
        protected_token_id = (
            token_ids["graph_start_token_ids"][views[0]]
            if views
            else token_ids["rec_token_id"]
        )
        truncated = [
            self._truncate(input_ids, prompt_length, protected_token_id)
            for input_ids, prompt_length in zip(tokenized_texts, full_prompt_lengths)
        ]
        retained_input_ids, retained_prompt_lengths = zip(*truncated)
        encoded = self.processor.tokenizer.pad(
            [{"input_ids": input_ids} for input_ids in retained_input_ids],
            padding=True,
            return_tensors="pt",
        )

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row_index, retained_prompt_length in enumerate(retained_prompt_lengths):
            labels[row_index, :retained_prompt_length] = -100

        for view in views:
            graphs: list[HypergraphData] = [
                graph_set[view] for graph_set in fitted_graphs
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

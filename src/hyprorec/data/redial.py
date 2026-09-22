"""Turn-level ReDial samples for independent recommendation and conversation tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ..configuration_hocrs import ALL_GRAPH_VIEWS
from ..processing_hocrs import HoCRSProcessor
from .batch import BatchData, batch_hypergraphs
from .hypergraph import HypergraphData, HypergraphTable


@dataclass
class HoCRSDatasetConfig:
    dataset_path: str
    task: str = "recommendation"
    hyperedge_table_path: str | None = None
    views: tuple[str, ...] = ALL_GRAPH_VIEWS
    topk: int = 3
    khop: int = 2
    sampling: str = "strict"
    sample_repeat: int = 1

    def __post_init__(self) -> None:
        if self.task not in {"recommendation", "conversation"}:
            raise ValueError("task must be recommendation or conversation.")
        if set(self.views) - set(ALL_GRAPH_VIEWS):
            raise ValueError("Unknown graph view.")
        if not 0 <= self.topk <= 10 or self.khop < 1:
            raise ValueError("topk must be in [0,10] and khop must be positive.")
        if self.sampling not in {"strict", "random"} or self.sample_repeat < 1:
            raise ValueError("Invalid sampling mode or sample repeat.")


class HoCRSDataset(Dataset):
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
        self.config = config
        self.split = split
        table_path = config.hyperedge_table_path or str(
            Path(config.dataset_path) / "hyperedge_table.json"
        )
        self.hypergraph_table = hypergraph_table or HypergraphTable.from_json(
            table_path
        )
        with (Path(config.dataset_path) / self.SPLIT_FILES[split]).open(
            encoding="utf-8"
        ) as file:
            self.samples = self._build_samples(json.load(file), config.task)
        self.repeat = config.sample_repeat if split == "train" else 1

    @staticmethod
    def _build_samples(
        conversations: Sequence[dict[str, Any]], task: str
    ) -> list[dict[str, Any]]:
        samples = []
        for conversation in conversations:
            context, context_items = [], []
            for turn in conversation["dialog"]:
                items = [int(item) for item in turn.get("items", [])]
                entry = {
                    "context": "\n".join(context),
                    "context_item_ids": list(dict.fromkeys(context_items)),
                    "response": turn["text"],
                    "role": turn["role"],
                }
                if task == "conversation":
                    samples.append(entry)
                else:
                    for item in items:
                        samples.append({**entry, "target_item_id": item})
                context.append(f"[{turn['role']}]: {turn['text']}")
                context_items.extend(items)
        return samples

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index // self.repeat]
        sampling = self.config.sampling if self.split == "train" else "strict"
        graphs = {}
        if sample["context_item_ids"]:
            graphs = {
                view: self.hypergraph_table.build_local(
                    sample["context_item_ids"],
                    view,
                    self.config.topk,
                    self.config.khop,
                    sampling,
                )
                for view in self.config.views
            }
        return {**sample, "hypergraphs": graphs}


class HoCRSDataCollator:
    def __init__(
        self, processor: HoCRSProcessor, task: str, max_history_tokens: int = 256
    ) -> None:
        self.processor = processor
        self.task = task
        self.max_history_tokens = max_history_tokens

    @staticmethod
    def _positions(input_ids, start_id, end_id, value_id, counts):
        positions = []
        for row, count in enumerate(counts):
            if count == 0:
                continue
            ids = input_ids[row].tolist()
            start = ids.index(start_id)
            end = ids.index(end_id, start)
            columns = [
                column for column in range(start + 1, end) if ids[column] == value_id
            ]
            positions.extend((row, column) for column in columns)
        return torch.tensor(positions, dtype=torch.long).reshape(-1, 2)

    def __call__(self, features: Sequence[dict[str, Any]]) -> BatchData:
        tokenizer = self.processor.tokenizer
        prompts = []
        prompt_lengths = []
        for sample in features:
            history = tokenizer(sample["context"], add_special_tokens=False)[
                "input_ids"
            ][-self.max_history_tokens :]
            context = tokenizer.decode(history, skip_special_tokens=False)
            sizes = {
                view: (graph.num_nodes, graph.num_hyperedges)
                for view, graph in sample["hypergraphs"].items()
            }
            prompt = self.processor.build_prompt(context, sizes)
            if self.task == "conversation":
                prompt += f"\n{sample['role']}:"
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = (
                tokenizer(
                    sample["response"] + (tokenizer.eos_token or ""),
                    add_special_tokens=False,
                )["input_ids"]
                if self.task == "conversation"
                else []
            )
            prompts.append({"input_ids": prompt_ids + response_ids})
            prompt_lengths.append(len(prompt_ids))
        encoded = tokenizer.pad(prompts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        token_ids = self.processor.get_token_id_map()
        hypergraphs = {}
        for view in dict.fromkeys(
            view for sample in features for view in sample["hypergraphs"]
        ):
            rows = [sample["hypergraphs"].get(view) for sample in features]
            graphs = [graph for graph in rows if graph is not None]
            graph = batch_hypergraphs(graphs)
            graph["node_positions"] = self._positions(
                input_ids,
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["node_token_id"],
                [entry.num_nodes if entry else 0 for entry in rows],
            )
            graph["hyperedge_positions"] = self._positions(
                input_ids,
                token_ids["graph_start_token_ids"][view],
                token_ids["graph_end_token_ids"][view],
                token_ids["hyperedge_token_id"],
                [entry.num_hyperedges if entry else 0 for entry in rows],
            )
            hypergraphs[view] = graph
        batch = BatchData(
            {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"],
                "hypergraphs": hypergraphs,
            }
        )
        if self.task == "recommendation":
            batch["pooling_mask"] = encoded["attention_mask"].bool()
            batch["rec_labels"] = torch.tensor(
                [sample["target_item_id"] for sample in features], dtype=torch.long
            )
        else:
            labels = input_ids.clone()
            labels[encoded["attention_mask"] == 0] = -100
            for row, length in enumerate(prompt_lengths):
                labels[row, :length] = -100
            batch["labels"] = labels
        return batch


__all__ = ["HoCRSDataCollator", "HoCRSDataset", "HoCRSDatasetConfig"]

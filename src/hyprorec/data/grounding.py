"""Catalogue-level batches for the HoCRS Grounding stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import csv
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

from .batch import BatchData, HypergraphBatch, batch_hypergraphs
from .hypergraph import HypergraphTable
from .source_catalogue import GroundingCatalogueDataset


class GroundingSourceDataset(GroundingCatalogueDataset):
    def __init__(self, dataset_path, views):
        super().__init__(dataset_path, views)
        with (Path(dataset_path) / "movies_info.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            self.texts = [row["movieName"] for row in csv.DictReader(file)]


class HoCRSGroundingDataset(Dataset):
    def __init__(
        self,
        hypergraph_table: HypergraphTable,
        views: Sequence[str],
        topk: int = 3,
        khop: int = 2,
        item_ids: Sequence[int] | None = None,
    ) -> None:
        self.hypergraph_table = hypergraph_table
        self.views = tuple(views)
        self.topk = topk
        self.khop = khop
        self.item_ids = list(item_ids or range(hypergraph_table.num_items))

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_id = self.item_ids[index]
        return {
            "item_id": item_id,
            "hypergraphs": {
                view: self.hypergraph_table.build_local(
                    [item_id],
                    "co" if view.startswith("co_") else view,
                    self.topk,
                    self.khop,
                )
                for view in self.views
            },
        }


class HoCRSGroundingCollator:
    def __init__(
        self,
        feature_tables: Mapping[str, torch.Tensor],
        views: Sequence[str],
        source_dataset: GroundingSourceDataset,
        tokenizer=None,
    ) -> None:
        self.feature_tables = feature_tables
        self.views = tuple(views)
        self.source_dataset = source_dataset
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def __call__(self, samples: Sequence[dict[str, Any]]) -> BatchData:
        hypergraphs: dict[str, HypergraphBatch] = {}
        node_features: dict[str, torch.Tensor] = {}
        source_data = {}
        for view in self.views:
            graphs = [sample["hypergraphs"][view] for sample in samples]
            batch = batch_hypergraphs(graphs)
            hypergraphs[view] = batch
            source_view = view.removeprefix("co_")
            node_features[view] = self.feature_tables[source_view].index_select(
                0, batch["node_ids"]
            )
            values = [
                self.source_dataset[node_id][source_view]
                for node_id in batch["node_ids"].tolist()
            ]
            if source_view == "txt":
                source_data[view] = dict(
                    self.tokenizer(
                        values,
                        padding=True,
                        return_tensors="pt",
                    )
                )
            else:
                values = torch.from_numpy(np.stack(values)).float()
                key = "input_values" if source_view == "ado" else "pixel_values"
                source_data[view] = {
                    key: values / (128.0 if source_view == "ado" else 255.0)
                }
        return BatchData(
            {
                "node_features": node_features,
                "hypergraphs": hypergraphs,
                "source_data": source_data,
                "return_loss": True,
            }
        )


__all__ = ["HoCRSGroundingCollator", "HoCRSGroundingDataset"]

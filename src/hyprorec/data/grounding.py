"""Catalogue-level batches for the HoCRS Grounding stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from .batch import BatchData, HypergraphBatch, batch_hypergraphs
from .hypergraph import HypergraphTable


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
                    [item_id], view, self.topk, self.khop
                )
                for view in self.views
            },
        }


class HoCRSGroundingCollator:
    def __init__(
        self,
        feature_tables: Mapping[str, torch.Tensor],
        views: Sequence[str],
    ) -> None:
        self.feature_tables = feature_tables
        self.views = tuple(views)

    def __call__(self, samples: Sequence[dict[str, Any]]) -> BatchData:
        hypergraphs: dict[str, HypergraphBatch] = {}
        node_features: dict[str, torch.Tensor] = {}
        for view in self.views:
            graphs = [sample["hypergraphs"][view] for sample in samples]
            batch = batch_hypergraphs(graphs)
            hypergraphs[view] = batch
            node_features[view] = self.feature_tables[view].index_select(
                0, batch["node_ids"]
            )
        return BatchData(
            {
                "node_features": node_features,
                "hypergraphs": hypergraphs,
                "return_loss": True,
                "labels": torch.zeros(len(samples), dtype=torch.long),
            }
        )


__all__ = ["HoCRSGroundingCollator", "HoCRSGroundingDataset"]

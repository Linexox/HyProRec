"""Hypergraph topology and local retrieval."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class HypergraphData:
    view: str
    node_ids: torch.Tensor
    hyperedge_index: torch.Tensor
    hyperedge_anchor_index: torch.Tensor

    @classmethod
    def from_hyperedges(cls, view, hyperedges):
        node_ids, node_to_local = [], {}
        incidence_nodes, incidence_edges, anchors = [], [], []

        def local_index(node_id):
            if node_id not in node_to_local:
                node_to_local[node_id] = len(node_ids)
                node_ids.append(node_id)
            return node_to_local[node_id]

        for edge_id, (anchor, neighbors) in enumerate(hyperedges):
            anchors.append(local_index(anchor))
            for node_id in dict.fromkeys((anchor, *neighbors)):
                incidence_nodes.append(local_index(node_id))
                incidence_edges.append(edge_id)
        return cls(
            view,
            torch.tensor(node_ids, dtype=torch.long),
            torch.tensor([incidence_nodes, incidence_edges], dtype=torch.long),
            torch.tensor(anchors, dtype=torch.long),
        )

    @property
    def num_nodes(self):
        return int(self.node_ids.numel())

    @property
    def num_hyperedges(self):
        return int(self.hyperedge_index[1].max()) + 1


class HypergraphTable:
    def __init__(self, tables):
        self.tables = tables
        self.num_items = len(next(iter(tables.values())))

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8") as file:
            return cls(json.load(file))

    def build_local(
        self,
        anchor_ids: int | Sequence[int],
        view: str,
        topk: int,
        khop: int,
        sampling: str = "strict",
        rng: random.Random | None = None,
    ):
        anchors = [anchor_ids] if isinstance(anchor_ids, int) else list(anchor_ids)
        frontier = list(dict.fromkeys(reversed(anchors)))
        visited, edges = set(), []
        for _ in range(khop):
            next_frontier = []
            for anchor in frontier:
                if anchor in visited:
                    continue
                row = self.tables[view][anchor]
                candidates = row[1:11]
                neighbors = (
                    candidates[:topk]
                    if sampling == "strict"
                    else (rng or random).sample(candidates, min(topk, len(candidates)))
                )
                edges.append((anchor, neighbors))
                visited.add(anchor)
                next_frontier.extend(neighbors)
            frontier = list(dict.fromkeys(next_frontier))
        return HypergraphData.from_hyperedges(view, edges)


__all__ = ["HypergraphData", "HypergraphTable"]

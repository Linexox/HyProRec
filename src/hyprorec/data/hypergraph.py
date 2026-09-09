"""Hypergraph topology primitives and local retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..constants import GRAPH_VIEWS


@dataclass(frozen=True)
class HypergraphData:
    """One local hypergraph with local incidence and global item ids."""

    view: str
    node_ids: torch.Tensor
    hyperedge_index: torch.Tensor
    # START: Preserve each hyperedge anchor for the v3 Grounding objective.
    hyperedge_anchor_index: torch.Tensor
    # END: Preserve each hyperedge anchor for the v3 Grounding objective.

    @classmethod
    def from_hyperedges(
        cls,
        view: str,
        hyperedges: Sequence[tuple[int, Sequence[int]]],
    ) -> HypergraphData:
        node_ids: list[int] = []
        node_to_local: dict[int, int] = {}
        incidence_nodes: list[int] = []
        incidence_edges: list[int] = []
        anchor_indices: list[int] = []

        def local_index(node_id: int) -> int:
            if node_id not in node_to_local:
                node_to_local[node_id] = len(node_ids)
                node_ids.append(node_id)
            return node_to_local[node_id]

        for edge_id, (anchor_id, neighbor_ids) in enumerate(hyperedges):
            members = list(dict.fromkeys((anchor_id, *neighbor_ids)))
            anchor_indices.append(local_index(anchor_id))
            for node_id in members:
                incidence_nodes.append(local_index(node_id))
                incidence_edges.append(edge_id)

        if not hyperedges:
            raise ValueError("A local hypergraph must contain at least one hyperedge.")
        return cls(
            view=view,
            node_ids=torch.tensor(node_ids, dtype=torch.long),
            hyperedge_index=torch.tensor(
                [incidence_nodes, incidence_edges],
                dtype=torch.long,
            ),
            hyperedge_anchor_index=torch.tensor(anchor_indices, dtype=torch.long),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.node_ids.numel())

    @property
    def num_hyperedges(self) -> int:
        return int(self.hyperedge_index[1].max().item()) + 1

    # START: Support whole-hyperedge removal during final prompt fitting.
    def truncate_hyperedges(self, count: int) -> HypergraphData:
        if not 1 <= count <= self.num_hyperedges:
            raise ValueError("count must retain at least one hyperedge.")
        if count == self.num_hyperedges:
            return self
        node_index, edge_index = self.hyperedge_index
        hyperedges = []
        for edge_id in range(count):
            members = self.node_ids[node_index[edge_index == edge_id]].tolist()
            anchor_id = int(self.node_ids[self.hyperedge_anchor_index[edge_id]])
            hyperedges.append((anchor_id, [item for item in members if item != anchor_id]))
        return HypergraphData.from_hyperedges(self.view, hyperedges)
    # END: Support whole-hyperedge removal during final prompt fitting.


class HypergraphTable:
    """Anchor-centered neighbor tables for co-occurrence and semantic views."""

    def __init__(self, tables: dict[str, list[list[int]]]) -> None:
        unknown_views = set(tables) - set(GRAPH_VIEWS)
        if unknown_views:
            raise ValueError(f"Unknown hypergraph table views: {sorted(unknown_views)}")
        if not tables:
            raise ValueError("At least one hypergraph view is required.")
        sizes = {len(rows) for rows in tables.values()}
        if len(sizes) != 1:
            raise ValueError("All hypergraph tables must contain the same items.")

        self.tables = tables
        self.num_items = sizes.pop()
        for view, rows in tables.items():
            for anchor_id, row in enumerate(rows):
                if not row or row[0] != anchor_id:
                    raise ValueError(
                        f"{view} row {anchor_id} must start with its anchor id."
                    )
                if any(node_id < 0 or node_id >= self.num_items for node_id in row):
                    raise ValueError(
                        f"{view} row {anchor_id} contains an invalid item id."
                    )

    @classmethod
    def from_json(cls, path: str | Path) -> HypergraphTable:
        with Path(path).open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise TypeError("hyperedge_table.json must contain an object.")
        return cls(payload)

    def build_local(
        self,
        anchor_ids: int | Sequence[int],
        view: str,
        topk: int,
        khop: int,
    ) -> HypergraphData:
        if view not in self.tables:
            raise KeyError(f"Missing hypergraph table for view '{view}'.")
        if topk < 0 or khop < 1:
            raise ValueError("topk must be non-negative and khop must be positive.")

        anchors = [anchor_ids] if isinstance(anchor_ids, int) else list(anchor_ids)
        assert anchors, "At least one anchor id is required."
        frontier = list(dict.fromkeys(reversed(anchors)))[:8]
        visited: set[int] = set()
        hyperedges: list[tuple[int, list[int]]] = []
        selected_nodes: set[int] = set()
        max_nodes = 120

        for _ in range(khop):
            next_frontier: list[int] = []
            for anchor_id in frontier:
                if anchor_id in visited: continue
                row = self.tables[view][anchor_id]
                neighbors = row[1 : topk + 1]
                candidate_nodes = {anchor_id, *neighbors}
                if len(selected_nodes | candidate_nodes) > max_nodes:
                    if not hyperedges:
                        raise ValueError("The first hyperedge exceeds the 120-node limit.")
                    return HypergraphData.from_hyperedges(view, hyperedges)
                selected_nodes.update(candidate_nodes)
                hyperedges.append((anchor_id, neighbors))
                visited.add(anchor_id)
                next_frontier.extend(neighbors)
            frontier = list(dict.fromkeys(next_frontier))
            if not frontier: break

        return HypergraphData.from_hyperedges(view, hyperedges)


__all__ = ["HypergraphData", "HypergraphTable"]

"""Tensor-only batch containers compatible with Transformers Trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import torch
from transformers import BatchFeature

from .hypergraph import HypergraphData


def _move_to_device(value: Any, device: torch.device | str, non_blocking: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return type(value)(
            {
                key: _move_to_device(item, device, non_blocking)
                for key, item in value.items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device, non_blocking) for item in value]
    return value


class BatchData(BatchFeature):
    """A mapping batch with recursive, non-mutating device transfer."""

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> BatchData:
        return type(self)(
            {
                key: _move_to_device(value, device, non_blocking)
                for key, value in self.items()
            }
        )


class HypergraphBatch(BatchData):
    """A disjoint union of same-view hypergraphs in one minibatch."""

    @property
    def batch_size(self) -> int:
        return int(self["node_ptr"].numel() - 1)

    @property
    def num_nodes(self) -> int:
        return int(self["node_ids"].numel())

    @property
    def num_hyperedges(self) -> int:
        return int(self["edge_ptr"][-1].item())


def batch_hypergraphs(graphs: Sequence[HypergraphData]) -> HypergraphBatch:
    """Pack variable-size graphs without adding cross-sample incidences."""

    if not graphs:
        raise ValueError("At least one hypergraph is required.")
    views = {graph.view for graph in graphs}
    if len(views) != 1:
        raise ValueError(
            f"A HypergraphBatch must contain one view, got {sorted(views)}."
        )

    node_ids = []
    incidence_indices = []
    # START: Offset local anchor indices together with packed node indices.
    anchor_indices = []
    # END: Offset local anchor indices together with packed node indices.
    node_ptr = [0]
    edge_ptr = [0]
    for graph in graphs:
        incidence = graph.hyperedge_index.clone()
        incidence[0] += node_ptr[-1]
        incidence[1] += edge_ptr[-1]
        node_ids.append(graph.node_ids)
        incidence_indices.append(incidence)
        anchor_indices.append(graph.hyperedge_anchor_index + node_ptr[-1])
        node_ptr.append(node_ptr[-1] + graph.num_nodes)
        edge_ptr.append(edge_ptr[-1] + graph.num_hyperedges)

    return HypergraphBatch(
        {
            "node_ids": torch.cat(node_ids),
            "hyperedge_index": torch.cat(incidence_indices, dim=1),
            "node_ptr": torch.tensor(node_ptr, dtype=torch.long),
            "edge_ptr": torch.tensor(edge_ptr, dtype=torch.long),
            "hyperedge_anchor_index": torch.cat(anchor_indices),
        }
    )


__all__ = ["BatchData", "HypergraphBatch", "batch_hypergraphs"]

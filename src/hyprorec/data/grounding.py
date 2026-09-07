"""Grounding samples built from the fixed item catalogue and hypergraph table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..arguments import GroundingArguments
from .batch import BatchData, HypergraphBatch, batch_hypergraphs
from .hypergraph import HypergraphData, HypergraphTable

_MM_PREFIX = {"img": "image", "ado": "audio", "vdo": "video"}


def _load_texts(dataset_path: Path) -> list[str]:
    with (dataset_path / "movies_info.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return [
            f"{row['movieName'].strip()} {row.get('description', '').strip()}".strip()
            for row in csv.DictReader(file)
        ]


class HoCRSGroundingDataset(Dataset):
    """One item anchor per sample; the local graph may contain multiple edges."""

    def __init__(self, config: GroundingArguments, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("Grounding split must be train or validation.")
        self.config = config
        self.split = split
        self.dataset_path = Path(config.dataset_path)
        self.texts = _load_texts(self.dataset_path)
        self.num_items = len(self.texts)
        table_path = config.hyperedge_table_path or (
            self.dataset_path / "hyperedge_table.json"
        )
        self.hypergraph_table = HypergraphTable.from_json(table_path)
        if self.hypergraph_table.num_items != self.num_items:
            raise ValueError("Hypergraph and catalogue contain different item counts.")
        self.graph_features = self._load_graph_features()
        self._mm_index = self._build_mm_index()
        generator = torch.Generator().manual_seed(config.seed)
        order = torch.randperm(self.num_items, generator=generator).tolist()
        split_at = max(1, int(self.num_items * (1.0 - config.validation_ratio)))
        self.anchor_ids = order[:split_at] if split == "train" else order[split_at:]
        if not self.anchor_ids:
            raise ValueError(f"Grounding {split} split is empty.")

    def _load_graph_features(self) -> dict[str, torch.Tensor]:
        directory = self.dataset_path / self.config.embeddings_dir_name
        result: dict[str, torch.Tensor] = {}
        for view in self.config.modalities:
            source_view = "txt" if view == "co" else view
            path = directory / f"{source_view}_embeddings.pt"
            table = torch.load(path, map_location="cpu", weights_only=True).float()
            if table.ndim != 2 or table.size(0) != self.num_items:
                raise ValueError(f"Invalid {source_view} embedding table: {path}")
            result[view] = table
        return result

    def _build_mm_index(self) -> dict[str, list[tuple[Path, int]]]:
        result: dict[str, list[tuple[Path, int]]] = {}
        for view in self.config.modalities:
            if view in {"co", "txt"}:
                continue
            paths = sorted(
                (self.dataset_path / "mm").glob(f"{_MM_PREFIX[view]}_*.npy"),
                key=lambda path: int(path.stem.rsplit("_", 1)[1]),
            )
            index: list[tuple[Path, int]] = []
            for path in paths:
                block = np.load(path, mmap_mode="r")
                index.extend((path, row) for row in range(block.shape[0]))
            if len(index) != self.num_items:
                raise ValueError(
                    f"Raw {view} data has {len(index)} items, "
                    f"expected {self.num_items}."
                )
            result[view] = index
        return result

    def _raw(self, view: str, node_ids: Sequence[int]) -> Any:
        if view in {"co", "txt"}:
            return [self.texts[node_id] for node_id in node_ids]
        values = []
        cache: dict[Path, np.ndarray] = {}
        for node_id in node_ids:
            path, row = self._mm_index[view][node_id]
            if path not in cache:
                cache[path] = np.load(path, mmap_mode="r")
            values.append(np.asarray(cache[path][row]))
        return np.stack(values)

    def __len__(self) -> int:
        return len(self.anchor_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_id = self.anchor_ids[index]
        graphs = {
            view: self.hypergraph_table.build_local(
                anchor_id, view, self.config.topk, self.config.khop
            )
            for view in self.config.modalities
        }
        return {"anchor_id": anchor_id, "hypergraphs": graphs}

    def collate_fn(self, samples: Sequence[dict[str, Any]]) -> BatchData:
        hypergraphs: dict[str, HypergraphBatch] = {}
        source_data: dict[str, Any] = {}
        for view in self.config.modalities:
            graphs = [sample["hypergraphs"][view] for sample in samples]
            batch = batch_hypergraphs(graphs)
            hypergraphs[view] = batch
            source_data[view] = self._raw(view, batch["node_ids"].tolist())
        return BatchData(
            {
                "anchor_ids": torch.tensor(
                    [sample["anchor_id"] for sample in samples], dtype=torch.long
                ),
                "hypergraphs": hypergraphs,
                "graph_features": self.graph_features,
                "source_data": source_data,
            }
        )


__all__ = ["HoCRSGroundingDataset"]

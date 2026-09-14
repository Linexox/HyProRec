"""Raw catalogue items used by the standalone Grounding stage."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import MODALITIES

MM_FILE_PREFIX = {"img": "image", "ado": "audio", "vdo": "video"}


def _load_texts(dataset_path: Path) -> list[str]:
    with (dataset_path / "movies_info.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return [
            f"{row['movieName'].strip()} {row.get('description', '').strip()}".strip()
            for row in csv.DictReader(file)
        ]


class GroundingCatalogueDataset(Dataset):
    """One catalogue item per sample, backed by memory-mapped modality blocks."""

    def __init__(
        self,
        dataset_path: str | Path,
        modalities: Sequence[str] = MODALITIES,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.modalities = tuple(modalities)
        self.texts = _load_texts(self.dataset_path)
        self._blocks: dict[Path, np.ndarray] = {}
        self._indices = {
            modality: self._build_index(modality)
            for modality in self.modalities
            if modality != "txt"
        }

    def _build_index(self, modality: str) -> list[tuple[Path, int]]:
        paths = sorted(
            (self.dataset_path / "mm").glob(f"{MM_FILE_PREFIX[modality]}_*.npy"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        index = []
        for path in paths:
            block = np.load(path, mmap_mode="r")
            index.extend((path, row) for row in range(len(block)))
        if len(index) != len(self.texts):
            raise ValueError(
                f"Raw {modality} data has {len(index)} items, expected {len(self.texts)}."
            )
        return index

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_id = index
        sample: dict[str, Any] = {"item_id": item_id}
        for modality in self.modalities:
            if modality == "txt":
                value = self.texts[item_id]
                present = bool(value.strip())
            else:
                path, row = self._indices[modality][item_id]
                if path not in self._blocks:
                    self._blocks[path] = np.load(path, mmap_mode="r")
                value = np.array(self._blocks[path][row], copy=True)
                present = bool(np.any(value))
            sample[modality] = value
            sample[f"{modality}_present"] = present
        return sample


__all__ = ["GroundingCatalogueDataset"]

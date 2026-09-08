"""Raw catalogue items and preprocessing for multimodal alignment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoTokenizer

from ..configuration_alignment import HoCRSAlignmentConfig
from ..constants import MODALITIES
from .batch import BatchData

MM_FILE_PREFIX = {"img": "image", "ado": "audio", "vdo": "video"}


def _load_texts(dataset_path: Path) -> list[str]:
    with (dataset_path / "movies_info.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return [
            f"{row['movieName'].strip()} {row.get('description', '').strip()}".strip()
            for row in csv.DictReader(file)
        ]


class HoCRSAlignmentDataset(Dataset):
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


class HoCRSAlignmentCollator:
    """Apply the pretrained source processors to one multimodal item batch."""

    def __init__(
        self, config: HoCRSAlignmentConfig, max_text_length: int = 128
    ) -> None:
        self.config = config
        self.max_text_length = max_text_length
        self.processors = {
            modality: (
                AutoTokenizer.from_pretrained(config.get_model_name_or_path(modality))
                if modality == "txt"
                else (
                    AutoFeatureExtractor.from_pretrained(
                        config.get_model_name_or_path(modality)
                    )
                    if modality == "ado"
                    else AutoImageProcessor.from_pretrained(
                        config.get_model_name_or_path(modality)
                    )
                )
            )
            for modality in config.modalities
        }

    def __call__(self, samples: Sequence[dict[str, Any]]) -> BatchData:
        batch: dict[str, Any] = {
            "item_ids": torch.tensor([sample["item_id"] for sample in samples]),
            "modality_mask": torch.tensor(
                [
                    [
                        sample[f"{modality}_present"]
                        for modality in self.config.modalities
                    ]
                    for sample in samples
                ],
                dtype=torch.bool,
            ),
        }
        for modality in self.config.modalities:
            values = [sample[modality] for sample in samples]
            if modality == "txt":
                encoded = self.processors[modality](
                    values,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
            elif modality == "ado":
                encoded = self.processors[modality](
                    [value.astype(np.float32) / 128.0 for value in values],
                    sampling_rate=16000,
                    padding=True,
                    return_tensors="pt",
                )
            elif modality == "img":
                encoded = self.processors[modality](images=values, return_tensors="pt")
            else:
                encoded = self.processors[modality](
                    [list(value) for value in values], return_tensors="pt"
                )
            for name, value in encoded.items():
                batch[f"{modality}_{name}"] = value
        return BatchData(batch)


def split_item_ids(
    num_items: int,
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")
    order = torch.randperm(num_items, generator=torch.Generator().manual_seed(seed))
    split_at = max(1, int(num_items * (1.0 - validation_ratio)))
    return order[:split_at].tolist(), order[split_at:].tolist()


__all__ = ["HoCRSAlignmentCollator", "HoCRSAlignmentDataset", "split_item_ids"]

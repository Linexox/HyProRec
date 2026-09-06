"""Create an offline fixed-slot multimodal content initialization table."""

# START: Build traceable Item Table and co-table initialization outside training.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..constants import MODALITIES


def build_content_table(
    embedding_tables: dict[str, torch.Tensor],
    enabled_modalities: list[str],
) -> torch.Tensor:
    enabled = set(enabled_modalities)
    unknown = enabled - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if not enabled:
        raise ValueError("At least one content modality must be enabled.")
    missing = set(MODALITIES) - set(embedding_tables)
    if missing:
        raise ValueError(f"Missing embedding tables: {sorted(missing)}")

    num_items = {table.size(0) for table in embedding_tables.values()}
    if len(num_items) != 1 or any(
        table.ndim != 2 for table in embedding_tables.values()
    ):
        raise ValueError("Embedding tables must be 2D and contain the same items.")

    slots = []
    for modality in MODALITIES:
        table = embedding_tables[modality].float()
        slots.append(
            F.normalize(table, dim=-1)
            if modality in enabled
            else torch.zeros_like(table)
        )
    return F.normalize(torch.cat(slots, dim=-1), dim=-1)


def prepare_content_table(
    embedding_dir: Path,
    output: Path,
    enabled_modalities: list[str],
) -> None:
    tables = {
        modality: torch.load(
            embedding_dir / f"{modality}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        for modality in MODALITIES
    }
    content = build_content_table(tables, enabled_modalities)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(content, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "enabled_modalities": enabled_modalities,
                "slot_order": list(MODALITIES),
                "source_shapes": {
                    modality: list(table.shape) for modality, table in tables.items()
                },
                "output_shape": list(content.shape),
                "normalized": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=MODALITIES,
        required=True,
    )
    args = parser.parse_args()
    prepare_content_table(args.embedding_dir, args.output, args.modality)


if __name__ == "__main__":
    main()
# END: Build traceable Item Table and co-table initialization outside training.

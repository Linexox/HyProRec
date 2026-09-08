"""Create an offline content table from aligned modality embeddings."""

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
    modality_mask: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    enabled = set(enabled_modalities)
    unknown = enabled - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if not enabled:
        raise ValueError("At least one content modality must be enabled.")
    missing = enabled - set(embedding_tables)
    if missing:
        raise ValueError(f"Missing embedding tables: {sorted(missing)}")

    selected_tables = {key: embedding_tables[key] for key in enabled_modalities}
    num_items = {table.size(0) for table in selected_tables.values()}
    if len(num_items) != 1 or any(
        table.ndim != 2 for table in selected_tables.values()
    ):
        raise ValueError("Embedding tables must be 2D and contain the same items.")
    widths = {table.size(1) for table in selected_tables.values()}
    if len(widths) != 1:
        raise ValueError("Aligned embedding tables must have the same width.")

    # START: Fuse aligned modalities by a masked spherical mean, not concatenation.
    count = next(iter(selected_tables.values())).new_zeros(
        (next(iter(num_items)), 1), dtype=torch.float32
    )
    content = next(iter(selected_tables.values())).new_zeros(
        (count.size(0), next(iter(widths))), dtype=torch.float32
    )
    for modality, table in selected_tables.items():
        valid = (
            modality_mask[modality].bool()
            if modality_mask is not None
            else torch.ones(table.size(0), dtype=torch.bool)
        )
        content[valid] += F.normalize(table[valid].float(), dim=-1)
        count[valid] += 1
    # START: Preserve a neutral zero row when an item lacks every enabled modality.
    return F.normalize(content / count.clamp_min(1), dim=-1)
    # END: Preserve a neutral zero row when an item lacks every enabled modality.
    # END: Fuse aligned modalities by a masked spherical mean, not concatenation.


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
        for modality in enabled_modalities
    }
    mask_path = embedding_dir / "modality_mask.pt"
    modality_mask = (
        torch.load(mask_path, map_location="cpu", weights_only=True)
        if mask_path.is_file()
        else None
    )
    content = build_content_table(tables, enabled_modalities, modality_mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(content, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "enabled_modalities": enabled_modalities,
                "fusion": "masked_normalized_mean",
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

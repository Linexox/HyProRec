"""Export normalized per-item tables from a trained alignment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.alignment import HoCRSAlignmentCollator, HoCRSAlignmentDataset
from ..modeling_alignment import HoCRSAlignmentModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = HoCRSAlignmentModel.from_pretrained(args.checkpoint).eval().to(device)
    dataset = HoCRSAlignmentDataset(args.dataset_dir, model.config.modalities)
    collator = HoCRSAlignmentCollator(model.config, args.max_text_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    tables = {modality: [] for modality in model.config.modalities}
    masks = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="aligned embeddings"):
            batch = batch.to(device)
            embeddings = model.encode(
                **{
                    key: value
                    for key, value in batch.items()
                    if key not in {"item_ids", "modality_mask"}
                }
            )
            batch_mask = batch["modality_mask"].cpu()
            for index, (modality, values) in enumerate(embeddings.items()):
                values = values.float().cpu()
                values[~batch_mask[:, index]] = 0
                tables[modality].append(values)
            masks.append(batch_mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for modality, chunks in tables.items():
        table = torch.cat(chunks)
        torch.save(table, args.output_dir / f"{modality}_embeddings.pt")
        shapes[modality] = list(table.shape)
    modality_mask = torch.cat(masks)
    mask_by_modality = {
        modality: modality_mask[:, index]
        for index, modality in enumerate(model.config.modalities)
    }
    torch.save(mask_by_modality, args.output_dir / "modality_mask.pt")
    (args.output_dir / "alignment_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "modalities": list(model.config.modalities),
                "shapes": shapes,
                "mask_shape": list(modality_mask.shape),
                "normalized": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

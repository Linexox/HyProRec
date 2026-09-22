"""Snapshot HoCRS2 catalogue/features and regenerate separated HyProRec edges."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..constants import MODALITIES
from .prepare_hyperedge_table import prepare_hyperedge_table


def prepare_dataset(source: Path, output: Path) -> None:
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "embeddings").mkdir()
    files = ["movies_info.csv", "train_data.json", "valid_data.json", "test_data.json"]
    files += [f"embeddings/{view}_embeddings.pt" for view in MODALITIES]
    provenance = {}
    for name in files:
        origin, target = source / name, output / name
        shutil.copy2(origin, target)
        source_hash = hashlib.sha256(origin.read_bytes()).hexdigest()
        provenance[name] = {"source": str(origin), "sha256": source_hash}
    with (output / "movies_info.csv").open(encoding="utf-8-sig", newline="") as stream:
        num_items = sum(1 for _ in csv.DictReader(stream))
    shapes = {}
    for view in MODALITIES:
        features = torch.load(
            output / f"embeddings/{view}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        shapes[view] = list(features.shape)
    prepare_hyperedge_table(
        output, output / "hyperedge_table.json", topk=10, device=torch.device("cpu")
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "hyprorec_revision": revision,
        "num_items": num_items,
        "embedding_shapes": shapes,
        "files": provenance,
        "topology_generator": "hyprorec.scripts.prepare_hyperedge_table",
        "topology_sha256": hashlib.sha256(
            (output / "hyperedge_table.json").read_bytes()
        ).hexdigest(),
        "topology_topk": 10,
        "semantic_edges": "cosine nearest neighbours per modality",
        "co_edges": "train conversations only",
    }
    (output / "provenance.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "num_items": num_items, "shapes": shapes}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.source, args.output)

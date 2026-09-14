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


# START: Keep transferred features and regenerated topology in one auditable snapshot.
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
        assert hashlib.sha256(target.read_bytes()).hexdigest() == source_hash
        provenance[name] = {"source": str(origin), "sha256": source_hash}
    with (output / "movies_info.csv").open(encoding="utf-8-sig", newline="") as stream:
        num_items = sum(1 for _ in csv.DictReader(stream))
    shapes = {}
    for view in MODALITIES:
        features = torch.load(output / f"embeddings/{view}_embeddings.pt", map_location="cpu", weights_only=True)
        if features.ndim != 2 or features.size(0) != num_items or not torch.isfinite(features).all():
            raise ValueError(f"Invalid {view} embedding table for {num_items} catalogue items.")
        shapes[view] = list(features.shape)
    for filename in files[1:4]:
        conversations = json.loads((output / filename).read_text(encoding="utf-8"))
        if any(not 0 <= int(item) < num_items for conversation in conversations
               for turn in conversation["dialog"] for item in turn.get("items", [])):
            raise ValueError(f"{filename} contains out-of-catalogue item IDs.")
    prepare_hyperedge_table(output, output / "hyperedge_table.json", topk=50, device=torch.device("cpu"))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "hyprorec_revision": revision, "num_items": num_items, "embedding_shapes": shapes,
        "files": provenance, "topology_generator": "hyprorec.scripts.prepare_hyperedge_table",
        "topology_sha256": hashlib.sha256((output / "hyperedge_table.json").read_bytes()).hexdigest(),
        "topology_topk": 50, "semantic_edges": "cosine nearest neighbours per modality",
        "cooccurrence_edges": "train dialogues only; separate co view",
        "crs_view": "vdo", "item_table": "random trainable; feature values are not used for its initialization",
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "num_items": num_items, "shapes": shapes}, indent=2))
# END: Keep transferred features and regenerated topology in one auditable snapshot.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.source, args.output)

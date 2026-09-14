"""Build separated co-occurrence and modality-similarity neighbor tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F

from ..constants import MODALITIES


def _load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def compute_similarity_neighbors(
    embeddings: torch.Tensor,
    topk: int,
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    """Compute top-k nearest neighbors for each item based on cosine similarity."""
    
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding table, got {embeddings.shape}.")

    embeddings = F.normalize(embeddings.float(), dim=-1).to(device)
    num_items = embeddings.size(0)
    candidate_count = min(topk + 1, num_items)
    neighbors: list[list[int]] = []
    for start in range(0, num_items, batch_size):
        similarities = embeddings[start : start + batch_size] @ embeddings.T
        indices = similarities.topk(candidate_count, dim=-1).indices.cpu().tolist()
        for offset, candidates in enumerate(indices):
            anchor_id = start + offset
            filtered_candidates = [node_id for node_id in candidates if node_id != anchor_id]
            neighbors.append(filtered_candidates[:topk])
    return neighbors


def compute_cooccurrence_neighbors(
    dataset_dir: Path,
    num_items: int,
) -> list[list[int]]:
    """Match HoCRS2 granularity: unique items co-mentioned in one train dialogue."""

    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for conversation in _load_json(dataset_dir / "train_data.json"):
        item_ids = {
            int(item_id)
            for utterance in conversation["dialog"]
            for item_id in utterance.get("items", [])
        }
        
        for left, right in combinations(sorted(item_ids), 2):
            counts[left][right] += 1
            counts[right][left] += 1
    return [
        [
            node_id
            for node_id, _ in sorted(
                counts[anchor_id].items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        for anchor_id in range(num_items)
    ]


def prepare_hyperedge_table(
    dataset_dir: Path,
    output: Path,
    topk: int = 50,
    similarity_batch_size: int = 512,
    device: torch.device | None = None,
    embedding_dir: Path | None = None,
) -> None:
    if topk < 0:
        raise ValueError("topk must be non-negative.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dir = embedding_dir or dataset_dir / "embeddings"
    embeddings = {
        modality: torch.load(
            embedding_dir / f"{modality}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        for modality in MODALITIES
    }
    num_items = embeddings[MODALITIES[0]].size(0)

    cooccurrence = compute_cooccurrence_neighbors(dataset_dir, num_items)
    table: dict[str, list[list[int]]] = {
        "co": [
            [anchor_id, *cooccurrence[anchor_id][:topk]]
            for anchor_id in range(num_items)
        ]
    }
    for modality in MODALITIES:
        neighbors = compute_similarity_neighbors(
            embeddings[modality],
            topk=topk,
            batch_size=similarity_batch_size,
            device=device,
        )
        table[modality] = [
            [anchor_id, *neighbors[anchor_id]] for anchor_id in range(num_items)
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(table, file, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separated HoCRS hypergraph neighbor tables."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_hyperedge_table(
        dataset_dir=args.dataset_dir,
        output=args.output or args.dataset_dir / "hyperedge_table.json",
        topk=args.topk,
        similarity_batch_size=args.similarity_batch_size,
        device=torch.device(args.device),
        embedding_dir=args.embedding_dir,
    )


if __name__ == "__main__":
    main()

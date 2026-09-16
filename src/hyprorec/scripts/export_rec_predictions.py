"""Export recommendation top-k rankings from a trained HoCRS checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from hyprorec.config import parse_experiment_args
from hyprorec.data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.metrics import recommendation_metrics
from hyprorec.modeling_hocrs import HoCRSModel
from hyprorec.processing_hocrs import HoCRSProcessor


SPLIT_FILES = {"valid": "valid_data.json", "test": "test_data.json"}
DATASET_SPLITS = {"valid": "validation", "test": "test"}


def build_recommendation_metadata(
    conversations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror ``HoCRSDataset._build_samples`` while retaining sample identity."""

    rows: list[dict[str, Any]] = []
    for conversation_index, conversation in enumerate(conversations):
        conversation_id = str(conversation.get("conv_id", conversation_index))
        history_item_ids: list[int] = []
        for turn_index, turn in enumerate(conversation["dialog"]):
            turn_item_ids = [int(item_id) for item_id in turn.get("items", [])]
            if turn["role"] == "Recommender" and turn_item_ids:
                unique_history = list(dict.fromkeys(history_item_ids))
                turn_id = turn.get("utt_id", turn_index)
                for target_index, target_item_id in enumerate(turn_item_ids):
                    rows.append(
                        {
                            "sample_id": (
                                f"{conversation_id}:{turn_id}:{target_index}"
                            ),
                            "user_id": conversation.get("user_id"),
                            "conversation_id": conversation_id,
                            "turn_id": turn_id,
                            "target": target_item_id,
                            "history_item_ids": unique_history,
                        }
                    )
            history_item_ids.extend(turn_item_ids)
    return rows


def load_metadata(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    with (dataset_path / SPLIT_FILES[split]).open(encoding="utf-8") as file:
        conversations = json.load(file)
    return build_recommendation_metadata(conversations)


@torch.inference_mode()
def export_split(
    model: HoCRSModel,
    dataset: HoCRSDataset,
    metadata: Sequence[dict[str, Any]],
    collator: HoCRSDataCollator,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(dataset) != len(metadata):
        raise RuntimeError(
            "Dataset and metadata sample counts differ: "
            f"{len(dataset)} != {len(metadata)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(**batch, use_cache=False)
        topk = min(50, output.rec_scores.size(-1))
        scores, indices = output.rec_scores.topk(topk, dim=-1)
        offset = len(rows)
        for row_index, (row_scores, row_indices) in enumerate(zip(scores, indices)):
            row = dict(metadata[offset + row_index])
            row["top50"] = [int(item_id) for item_id in row_indices.cpu().tolist()]
            row["top50_scores"] = [
                float(score) for score in row_scores.float().cpu().tolist()
            ]
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics: dict[str, Any] = {
        "samples": len(rows),
        "source": str(output_path.resolve()),
    }
    metrics.update(
        recommendation_metrics(
            [row["top50"] for row in rows],
            [row["target"] for row in rows],
        )
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=tuple(SPLIT_FILES),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    model_args, data_args, training_args, _ = parse_experiment_args(
        ["--config", str(args.config.resolve())]
    )
    set_seed(training_args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer_path = (
        checkpoint.parent
        if (checkpoint.parent / "tokenizer.json").exists()
        else Path(model_args.backbone_name_or_path)
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        load_kwargs["device_map"] = {
            "": device.index if device.index is not None else 0
        }
    model = HoCRSModel.from_pretrained(str(checkpoint), **load_kwargs)
    if device.type != "cuda":
        model.to(device)

    processor = HoCRSProcessor(
        tokenizer=tokenizer,
        num_soft_prompt_tokens=model.config.num_soft_prompt_tokens,
        use_context_token=model.config.use_context_token,
    )
    collator = HoCRSDataCollator(
        processor,
        max_length=data_args.max_length,
        max_history_tokens=data_args.max_history_tokens,
        max_response_tokens=data_args.max_response_tokens,
    )
    dataset_path = Path(data_args.dataset_path)
    table_path = Path(
        data_args.hyperedge_table_path
        or dataset_path / "hyperedge_table.json"
    )
    hypergraph_table = (
        HypergraphTable.from_json(table_path) if data_args.views else None
    )
    dataset_config = HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        hyperedge_table_path=data_args.hyperedge_table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
    )
    output_dir = (args.output_dir or checkpoint / "predictions").resolve()
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "splits": {},
    }
    for split in args.splits:
        dataset = HoCRSDataset(
            dataset_config,
            DATASET_SPLITS[split],
            hypergraph_table,
        )
        summary["splits"][split] = export_split(
            model=model,
            dataset=dataset,
            metadata=load_metadata(dataset_path, split),
            collator=collator,
            output_path=output_dir / f"{split}_top50.json",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rec_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

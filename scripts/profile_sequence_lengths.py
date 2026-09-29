"""Profile HoCRS prompt lengths and local hypergraph sizes for one config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from hyprorec.config import parse_experiment_args
from hyprorec.data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.processing_hocrs import HoCRSProcessor


def _quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    model_args, data_args, _, _ = parse_experiment_args(["--config", str(args.config)])
    processor = HoCRSProcessor(
        AutoTokenizer.from_pretrained(model_args.backbone_name_or_path),
        model_args.num_prompt_tokens,
    )
    table_path = data_args.hyperedge_table_path or str(
        Path(data_args.dataset_path) / "hyperedge_table.json"
    )
    table = HypergraphTable.from_json(table_path)
    dataset_config = HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        task=model_args.task,
        hyperedge_table_path=table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
        max_hypergraph_nodes=data_args.max_hypergraph_nodes,
    )
    collator = HoCRSDataCollator(
        processor,
        model_args.task,
        data_args.max_history_tokens,
        data_args.max_sequence_tokens,
    )
    report: dict[str, object] = {
        "config": str(args.config),
        "max_sequence_tokens": data_args.max_sequence_tokens,
        "max_history_tokens": data_args.max_history_tokens,
        "max_hypergraph_nodes": data_args.max_hypergraph_nodes,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        dataset = HoCRSDataset(dataset_config, split, table)
        lengths: list[int] = []
        contexts: list[int] = []
        nodes = {view: [] for view in data_args.views}
        edges = {view: [] for view in data_args.views}
        over_limit = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            try:
                batch = collator([sample])
                length = int(batch["input_ids"].shape[1])
            except ValueError as error:
                if "max_sequence_tokens" not in str(error):
                    raise
                over_limit += 1
                tokenizer = processor.tokenizer
                history = tokenizer(sample["context"], add_special_tokens=False)[
                    "input_ids"
                ][-data_args.max_history_tokens :]
                context = tokenizer.decode(history, skip_special_tokens=False)
                sizes = {
                    view: (graph.num_nodes, graph.num_hyperedges)
                    for view, graph in sample["hypergraphs"].items()
                }
                prompt = processor.build_prompt(context, sizes)
                length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            lengths.append(length)
            contexts.append(
                len(
                    processor.tokenizer(sample["context"], add_special_tokens=False)[
                        "input_ids"
                    ][-data_args.max_history_tokens :]
                )
            )
            for view in data_args.views:
                graph = sample["hypergraphs"].get(view)
                nodes[view].append(graph.num_nodes if graph else 0)
                edges[view].append(graph.num_hyperedges if graph else 0)
        report["splits"][split] = {
            "sequence_tokens": _quantiles(lengths),
            "history_tokens": _quantiles(contexts),
            "over_limit": over_limit,
            "views": {
                view: {
                    "nodes": _quantiles(values),
                    "hyperedges": _quantiles(edges[view]),
                }
                for view, values in nodes.items()
            },
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

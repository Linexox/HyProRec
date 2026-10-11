"""Run one text-plus-graph CRS forward and cached generation per backbone."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from hyprorec.backbones import load_backbone_adapter
from hyprorec.configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from hyprorec.modeling_hocrs import HoCRSRecommendationModel
from hyprorec.processing_hocrs import HoCRSProcessor


DEFAULT_MODELS = {
    "qwen-omni-3b": "/data2/lhf/.MODEL/Qwen2.5-Omni-3B",
    "interactive-omni-4b": (
        "/data2/lhf/huggingface/hub/models--sensenova--InteractiveOmni-4B"
        "/snapshots/2460f0dc81df8527ec18e7e42e04b0031131d76d"
    ),
    "qwen-vl-3b": (
        "/data2/lhf/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
        "/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
    ),
    "qwen-text-3b": (
        "/data2/lhf/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
        "/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    ),
    "llava-7b": (
        "/data2/lhf/huggingface/hub/models--llava-hf--llava-1.5-7b-hf"
        "/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    ),
}


def _graph_inputs(processor, tokenizer, device):
    prompt = processor.build_prompt("Recommend one movie.", {"txt": (1, 1)})
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    ids = processor.get_token_id_map()
    node = torch.nonzero(input_ids == ids["node_token_id"], as_tuple=False)[0]
    edge = torch.nonzero(
        input_ids == ids["hyperedge_token_id"], as_tuple=False
    )[0]
    graph = {
        "node_ids": torch.zeros(1, dtype=torch.long, device=device),
        "hyperedge_index": torch.tensor(
            [[0], [0]], dtype=torch.long, device=device
        ),
        "edge_ptr": torch.tensor([0, 1], dtype=torch.long, device=device),
        "node_positions": node.reshape(1, 2),
        "hyperedge_positions": edge.reshape(1, 2),
    }
    return input_ids, attention_mask, graph


def run_one(name: str, path: str, device: torch.device) -> dict[str, object]:
    adapter = load_backbone_adapter(path)
    tokenizer = adapter.tokenizer
    processor = HoCRSProcessor(tokenizer, num_prompt_tokens=20)
    adapter.resize_token_embeddings(len(tokenizer))
    graph_config = HoCRSHypergraphConfig(
        input_dim=4,
        hidden_dim=4,
        output_dim=2,
        num_layers=1,
    )
    config = HoCRSConfig(
        backbone_config=adapter.config,
        views=("txt",),
        num_items=2,
        item_feature_dims={"txt": 4},
        item_table_view="txt",
        txt_hypergraph_config=graph_config,
        prompt_token_id=processor.get_token_id_map()["soft_prompt_token_id"],
        freeze_backbone=True,
    )
    model = HoCRSRecommendationModel(
        config,
        backbone_adapter=adapter,
    ).to(device).eval()
    model.initialize_feature_tables({"txt": torch.randn(2, 4)})
    input_ids, attention_mask, graph = _graph_inputs(processor, tokenizer, device)
    hypergraphs = {"txt": graph}
    with torch.inference_mode():
        encoded = model.encode(
            input_ids,
            attention_mask=attention_mask,
            hypergraphs=hypergraphs,
            use_cache=True,
        )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=attention_mask.bool(),
            rec_labels=torch.zeros(1, dtype=torch.long, device=device),
            hypergraphs=hypergraphs,
            use_cache=True,
        )
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hypergraphs=hypergraphs,
            max_new_tokens=1,
            use_cache=True,
        )
    result = {
        "name": name,
        "model_type": adapter.config.model_type,
        "hidden_size": adapter.hidden_size,
        "input_shape": list(input_ids.shape),
        "hidden_shape": list(output.hidden_states[-1].shape),
        "recommendation_shape": list(output.recommendation_logits.shape),
        "conversation_shape": list(output.conversation_logits.shape),
        "generated_shape": list(generated.shape),
        "cache_present": encoded.past_key_values is not None,
    }
    del model, adapter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*DEFAULT_MODELS, "all"], default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    names = DEFAULT_MODELS if args.model == "all" else {args.model: DEFAULT_MODELS[args.model]}
    results = [run_one(name, path, device) for name, path in names.items()]
    payload = {"results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

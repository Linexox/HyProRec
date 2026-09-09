"""Train the lightweight source and hypergraph towers for Grounding."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import Trainer

from ..arguments import DataArguments, ModelArguments
from ..config import parse_experiment_args
from ..configuration_grounding import HoCRSGroundingConfig
from ..constants import MODALITIES
from ..data.grounding import HoCRSGroundingCollator, HoCRSGroundingDataset
from ..data.hypergraph import HypergraphTable
from ..modeling_grounding import HoCRSGroundingModel


def _load_feature_tables(data_args: DataArguments) -> dict[str, torch.Tensor]:
    embedding_dir = Path(data_args.dataset_path) / data_args.embeddings_dir_name
    return {
        view: torch.load(
            embedding_dir / f"{view}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for view in data_args.views
        if view in MODALITIES
    }


def main() -> None:
    load_dotenv()
    model_args, data_args, training_args, config_path = parse_experiment_args()
    views = tuple(view for view in data_args.views if view in MODALITIES)
    feature_tables = _load_feature_tables(data_args)
    input_dims = {view: feature_tables[view].size(1) for view in views}
    config = HoCRSGroundingConfig(
        views=views,
        input_dims=input_dims,
        hidden_dim=model_args.hypergraph_hidden_dim,
        output_dim=model_args.hypergraph_output_dim,
        num_layers=model_args.hypergraph_num_layers,
        dropout=model_args.hypergraph_dropout,
        temperature=model_args.grounding_temperature,
        lambda_ga_sa=model_args.grounding_ga_sa_weight,
        lambda_ga_sn=model_args.grounding_ga_sn_weight,
        lambda_sa_sn=model_args.grounding_sa_sn_weight,
    )
    model = HoCRSGroundingModel(config)
    table_path = data_args.hyperedge_table_path or (
        Path(data_args.dataset_path) / "hyperedge_table.json"
    )
    table = HypergraphTable.from_json(table_path)
    dataset = HoCRSGroundingDataset(
        table,
        views,
        topk=data_args.topk,
        khop=data_args.khop,
    )
    split = max(1, int(len(dataset) * 0.9))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        [split, len(dataset) - split],
        generator=torch.Generator().manual_seed(training_args.seed),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=HoCRSGroundingCollator(feature_tables, views),
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )
    if trainer.is_world_process_zero():
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"grounding_recipe{config_path.suffix}").write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (output_dir / "resolved_grounding_config.json").write_text(
            json.dumps(
                {
                    "model": asdict(model_args),
                    "data": asdict(data_args),
                    "training": training_args.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    if training_args.do_train:
        result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

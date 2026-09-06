"""Train or evaluate HoCRS with the Transformers Trainer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint

from ..arguments import DataArguments, ModelArguments
from ..config import parse_experiment_args
from ..configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from ..constants import MODALITIES
from ..data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from ..data.hypergraph import HypergraphTable
from ..metrics import build_compute_metrics, preprocess_logits_for_metrics
from ..modeling_hocrs import HoCRSModel
from ..processing_hocrs import HoCRSProcessor


def _load_modality_tables(data_args: DataArguments) -> dict[str, torch.Tensor]:
    dataset_path = Path(data_args.dataset_path)
    embedding_path = dataset_path / data_args.embeddings_dir_name
    tables = {
        modality: torch.load(
            embedding_path / f"{modality}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for modality in MODALITIES
    }
    shapes = {tuple(table.shape) for table in tables.values()}
    if len(shapes) != 1:
        raise ValueError(f"Modality embedding shapes do not match: {sorted(shapes)}")
    if any(table.ndim != 2 for table in tables.values()):
        raise ValueError("Every modality embedding table must be two-dimensional.")
    return tables


def _build_feature_tables(
    modality_tables: dict[str, torch.Tensor],
    views: list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Build graph tables and an independent Item Table content initialization."""

    semantic_views = [view for view in views if view in MODALITIES]
    content_modalities = semantic_views or list(MODALITIES)
    normalized = [
        F.normalize(modality_tables[view], dim=-1) for view in content_modalities
    ]
    content_base = F.normalize(torch.stack(normalized).mean(dim=0), dim=-1)
    feature_tables = {view: modality_tables[view] for view in semantic_views}
    return feature_tables, content_base


def _build_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor: HoCRSProcessor,
    modality_tables: dict[str, torch.Tensor],
) -> HoCRSModel:
    backbone = AutoModelForCausalLM.from_pretrained(model_args.backbone_name_or_path)
    backbone.resize_token_embeddings(len(processor.tokenizer))

    token_ids = processor.get_token_id_map()
    graph_config = HoCRSHypergraphConfig(
        input_dim=next(iter(modality_tables.values())).size(1),
        hidden_dim=model_args.hypergraph_hidden_dim,
        output_dim=model_args.hypergraph_output_dim,
        num_layers=model_args.hypergraph_num_layers,
        dropout=model_args.hypergraph_dropout,
    )
    config = HoCRSConfig(
        backbone_config=backbone.config,
        views=data_args.views,
        num_items=next(iter(modality_tables.values())).size(0),
        item_dim=next(iter(modality_tables.values())).size(1),
        use_hypergraph_encoder=model_args.use_hypergraph_encoder,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_dropout=model_args.recommendation_dropout,
        recommendation_temperature=model_args.recommendation_temperature,
        beta=model_args.beta,
        num_soft_prompt_tokens=model_args.num_soft_prompt_tokens,
        freeze_backbone=model_args.freeze_backbone,
        train_special_tokens=model_args.train_special_tokens,
        co_hypergraph_config=graph_config,
        txt_hypergraph_config=graph_config,
        img_hypergraph_config=graph_config,
        ado_hypergraph_config=graph_config,
        vdo_hypergraph_config=graph_config,
        **token_ids,
    )
    model = HoCRSModel(config, backbone=backbone)
    feature_tables, item_table_init = _build_feature_tables(
        modality_tables, data_args.views
    )
    model.initialize_feature_tables(
        feature_tables,
        item_table_init=item_table_init,
    )
    return model


def _resolve_resume_checkpoint(training_args) -> str | None:
    if training_args.resume_from_checkpoint:
        return training_args.resume_from_checkpoint
    output_dir = Path(training_args.output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


def _save_experiment_provenance(
    output_dir: str,
    config_path: Path,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = config_path.suffix.lower()
    (destination / f"source_experiment_config{suffix}").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    resolved = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "training": training_args.to_dict(),
    }
    (destination / "resolved_experiment_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    model_args, data_args, training_args, config_path = parse_experiment_args()

    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(
        tokenizer=tokenizer,
        num_soft_prompt_tokens=model_args.num_soft_prompt_tokens,
    )
    modality_tables = _load_modality_tables(data_args)
    model = _build_model(model_args, data_args, processor, modality_tables)

    dataset_config = HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        hyperedge_table_path=data_args.hyperedge_table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
    )
    hypergraph_table = None
    if data_args.views:
        table_path = dataset_config.hyperedge_table_path or (
            Path(dataset_config.dataset_path) / "hyperedge_table.json"
        )
        hypergraph_table = HypergraphTable.from_json(table_path)
        if hypergraph_table.num_items != model.config.num_items:
            raise ValueError(
                "The hypergraph table and modality embeddings contain different item counts: "
                f"{hypergraph_table.num_items} and {model.config.num_items}."
            )

    train_dataset = (
        HoCRSDataset(dataset_config, "train", hypergraph_table)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        HoCRSDataset(dataset_config, "validation", hypergraph_table)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        HoCRSDataset(dataset_config, "test", hypergraph_table)
        if training_args.do_predict
        else None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=HoCRSDataCollator(processor, max_length=data_args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
        _save_experiment_provenance(
            training_args.output_dir,
            config_path,
            model_args,
            data_args,
            training_args,
        )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=_resolve_resume_checkpoint(training_args)
        )
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if test_dataset is not None:
        prediction = trainer.predict(test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", prediction.metrics)
        trainer.save_metrics("test", prediction.metrics)


if __name__ == "__main__":
    main()

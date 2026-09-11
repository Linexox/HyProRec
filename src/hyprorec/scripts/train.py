"""Train or evaluate HoCRS with the Transformers Trainer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, Trainer, set_seed  # Modified: seed before constructing modules.
from transformers.trainer_utils import get_last_checkpoint

from ..arguments import DataArguments, ModelArguments
from ..config import parse_experiment_args
from ..configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from ..constants import MODALITIES
from ..data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from ..data.hypergraph import HypergraphTable
from ..metrics import build_compute_metrics, preprocess_logits_for_metrics
from ..modeling_hocrs import HoCRSModel, load_backbone  # Modified: support Omni Thinker.
from ..processing_hocrs import HoCRSProcessor


def _load_modality_tables(data_args: DataArguments) -> dict[str, torch.Tensor]:
    # START: Load only enabled semantic views and retain their native widths.
    dataset_path = Path(data_args.dataset_path)
    embedding_path = dataset_path / data_args.embeddings_dir_name
    tables = {
        modality: torch.load(
            embedding_path / f"{modality}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for modality in data_args.views
        if modality in MODALITIES
    }
    if any(table.ndim != 2 for table in tables.values()):
        raise ValueError("Every modality embedding table must be two-dimensional.")
    # END: Load only enabled semantic views and retain their native widths.
    return tables


def _load_content_table(data_args: DataArguments) -> torch.Tensor:
    table = torch.load(
        data_args.content_table_path,
        map_location="cpu",
        weights_only=True,
    ).float()
    if table.ndim != 2:
        raise ValueError("The content initialization table must be two-dimensional.")
    return table


def _build_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor: HoCRSProcessor,
    modality_tables: dict[str, torch.Tensor],
    content_table: torch.Tensor,
) -> HoCRSModel:
    backbone = load_backbone(model_args.backbone_name_or_path)  # Modified: select Thinker for Omni.
    backbone.resize_token_embeddings(len(processor.tokenizer))

    token_ids = processor.get_token_id_map()
    num_items, item_dim = content_table.shape
    if any(table.size(0) != num_items for table in modality_tables.values()):
        raise ValueError("Content and modality tables contain different item counts.")
    view_input_dims = {
        **{view: table.size(1) for view, table in modality_tables.items()},
        **({"co": item_dim} if "co" in data_args.views else {}),
    }
    graph_configs = {
        view: HoCRSHypergraphConfig(
            input_dim=view_input_dims[view],
            hidden_dim=model_args.hypergraph_hidden_dim,
            output_dim=model_args.hypergraph_output_dim,
            num_layers=model_args.hypergraph_num_layers,
            dropout=model_args.hypergraph_dropout,
        )
        for view in data_args.views
    }
    config = HoCRSConfig(
        backbone_config=backbone.config,
        views=data_args.views,
        num_items=num_items,
        item_dim=(                                                          # ***** FIXME *****
            model_args.recommendation_hidden_dim
            if model_args.item_table_init == "random"
            else item_dim
        ),
        use_hypergraph_encoder=model_args.use_hypergraph_encoder,
        grounding_checkpoint_path=model_args.grounding_checkpoint_path,
        use_source_projector=model_args.use_source_projector,
        grounding_weight=model_args.grounding_weight,
        grounding_ga_sa_weight=model_args.grounding_ga_sa_weight,
        grounding_ga_sn_weight=model_args.grounding_ga_sn_weight,
        grounding_sa_sn_weight=model_args.grounding_sa_sn_weight,
        grounding_temperature=model_args.grounding_temperature,
        item_table_init=model_args.item_table_init,
        train_item_table=model_args.train_item_table,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_dropout=model_args.recommendation_dropout,
        recommendation_temperature=model_args.recommendation_temperature,
        beta=model_args.beta,
        num_soft_prompt_tokens=model_args.num_soft_prompt_tokens,
        freeze_backbone=model_args.freeze_backbone,
        train_special_tokens=model_args.train_special_tokens,
        co_hypergraph_config=graph_configs.get("co"),
        txt_hypergraph_config=graph_configs.get("txt"),
        img_hypergraph_config=graph_configs.get("img"),
        ado_hypergraph_config=graph_configs.get("ado"),
        vdo_hypergraph_config=graph_configs.get("vdo"),
        **token_ids,
    )
    model = HoCRSModel(config, backbone=backbone)
    if model_args.grounding_checkpoint_path:
        model.load_grounding_checkpoint(model_args.grounding_checkpoint_path)
    feature_tables = dict(modality_tables)
    if "co" in data_args.views:
        feature_tables["co"] = content_table
    model.initialize_feature_tables(
        feature_tables,
        item_table_init=(
            content_table if model_args.item_table_init == "aligned_content" else None
        ),
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
    set_seed(training_args.seed)  # Modified: cover all newly initialized parameters.

    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(
        tokenizer=tokenizer,
        num_soft_prompt_tokens=model_args.num_soft_prompt_tokens,
    )
    # START: Load prepared features without performing online fusion.
    modality_tables = _load_modality_tables(data_args)
    content_table = _load_content_table(data_args)
    model = _build_model(
        model_args,
        data_args,
        processor,
        modality_tables,
        content_table,
    )
    # END: Load prepared features without performing online fusion.

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
        # START: Pass explicit history and response budgets to the CRS collator.
        data_collator=HoCRSDataCollator(
            processor,
            max_length=data_args.max_length,
            max_history_tokens=data_args.max_history_tokens,
            max_response_tokens=data_args.max_response_tokens,
        ),
        # END: Pass explicit history and response budgets to the CRS collator.
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

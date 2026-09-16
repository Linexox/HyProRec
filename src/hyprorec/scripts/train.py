"""Train or evaluate HoCRS with the Transformers Trainer."""

from __future__ import annotations

import json
from copy import copy
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, Trainer, TrainerCallback, set_seed  # Modified: seed before constructing modules.
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
    return tables


def _load_item_target_frequencies(
    data_args: DataArguments,
    num_items: int,
) -> torch.Tensor:
    frequencies = torch.zeros(num_items, dtype=torch.float32)
    path = Path(data_args.dataset_path) / "train_data.json"
    with path.open(encoding="utf-8") as file:
        conversations = json.load(file)
    for conversation in conversations:
        for turn in conversation["dialog"]:
            if turn["role"] != "Recommender":
                continue
            for item_id in turn.get("items", []):
                item_id = int(item_id)
                if not 0 <= item_id < num_items:
                    raise ValueError(f"Train target item {item_id} is outside the catalogue.")
                frequencies[item_id] += 1
    return frequencies


def _build_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor: HoCRSProcessor,
    modality_tables: dict[str, torch.Tensor],
    item_frequencies: torch.Tensor,
) -> HoCRSModel:
    backbone = load_backbone(model_args.backbone_name_or_path)  # Modified: select Thinker for Omni.
    backbone.resize_token_embeddings(len(processor.tokenizer))

    token_ids = processor.get_token_id_map()
    if not modality_tables:
        raise ValueError("At least one modality table is required.")
    num_items = next(iter(modality_tables.values())).size(0)
    if any(table.size(0) != num_items for table in modality_tables.values()):
        raise ValueError("Modality tables contain different item counts.")
    view_input_dims = {
        **{view: table.size(1) for view, table in modality_tables.items()},
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
        item_dim=(
            model_args.semantic_item_dim
            if model_args.item_table_mode == "semantic_hybrid"
            else model_args.recommendation_hidden_dim
        ),
        use_hypergraph_encoder=model_args.use_hypergraph_encoder,
        grounding_checkpoint_path=model_args.grounding_checkpoint_path,
        item_table_mode=model_args.item_table_mode,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_dropout=model_args.recommendation_dropout,
        recommendation_temperature=model_args.recommendation_temperature,
        use_moe=model_args.use_moe,
        moe_num_experts=model_args.moe_num_experts,
        moe_hidden_dim=model_args.moe_hidden_dim,
        moe_router_temperature=model_args.moe_router_temperature,
        moe_residual_scale_init=model_args.moe_residual_scale_init,
        beta=model_args.beta,
        num_soft_prompt_tokens=model_args.num_soft_prompt_tokens,
        use_context_token=model_args.use_context_token,
        freeze_backbone=model_args.freeze_backbone,
        train_special_tokens=model_args.train_special_tokens,
        txt_hypergraph_config=graph_configs.get("txt"),
        img_hypergraph_config=graph_configs.get("img"),
        ado_hypergraph_config=graph_configs.get("ado"),
        vdo_hypergraph_config=graph_configs.get("vdo"),
        **token_ids,
    )
    model = HoCRSModel(config, backbone=backbone)
    if model_args.grounding_checkpoint_path:
        model.load_grounding_checkpoint(model_args.grounding_checkpoint_path)
    model.initialize_feature_tables(
        modality_tables,
        item_frequencies=item_frequencies,
    )
    return model


def _resolve_resume_checkpoint(training_args) -> str | None:
    if training_args.resume_from_checkpoint:
        return training_args.resume_from_checkpoint
    output_dir = Path(training_args.output_dir)
    if not output_dir.is_dir():
        return None
    return get_last_checkpoint(str(output_dir))


class TestEvaluationCallback(TrainerCallback):
    """Run held-out test evaluation after each completed training epoch."""

    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        training_control = copy(control)
        metrics = self.trainer.evaluate(
            eval_dataset=self.test_dataset,
            metric_key_prefix="test",
        )
        if self.trainer.is_world_process_zero():
            self.trainer.log_metrics("test", metrics)
            self.trainer.save_metrics("test", metrics)
        self.trainer.control = training_control
        return training_control


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
        use_context_token=model_args.use_context_token,
    )
    modality_tables = _load_modality_tables(data_args)
    if not modality_tables:
        raise ValueError("CRS training requires at least one modality view.")
    item_frequencies = _load_item_target_frequencies(
        data_args,
        next(iter(modality_tables.values())).size(0),
    )
    model = _build_model(
        model_args,
        data_args,
        processor,
        modality_tables,
        item_frequencies,
    )

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

    train_dataset = HoCRSDataset(dataset_config, "train", hypergraph_table) if training_args.do_train else None
    eval_dataset = HoCRSDataset(dataset_config, "validation", hypergraph_table) if training_args.do_eval else None
    test_dataset = HoCRSDataset(dataset_config, "test", hypergraph_table) if (training_args.do_train or training_args.do_predict) else None

    test_callback = TestEvaluationCallback(test_dataset) if test_dataset is not None else None
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=HoCRSDataCollator(
            processor,
            max_length=data_args.max_length,
            max_history_tokens=data_args.max_history_tokens,
            max_response_tokens=data_args.max_response_tokens,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[test_callback] if test_callback is not None else None,
    )
    if test_callback is not None:
        test_callback.trainer = trainer

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
        train_result = trainer.train(resume_from_checkpoint=_resolve_resume_checkpoint(training_args))
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

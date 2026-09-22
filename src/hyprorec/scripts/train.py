"""Train one independent HyProRec task per invocation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, Trainer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from ..arguments import DataArguments, ModelArguments
from ..config import parse_experiment_args
from ..configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from ..constants import MODALITIES
from ..data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from ..data.hypergraph import HypergraphTable
from ..metrics import build_compute_metrics, preprocess_logits_for_metrics
from ..modeling_hocrs import (
    HoCRSConversationModel,
    HoCRSRecommendationModel,
    load_backbone,
)
from ..processing_hocrs import HoCRSProcessor


def _load_modality_tables(
    data_args: DataArguments, model_args: ModelArguments
) -> dict[str, torch.Tensor]:
    graph_views = {view for view in data_args.views if view in MODALITIES}
    if "co" in data_args.views:
        graph_views.add(model_args.co_feature_view)
    item_views = (
        set(MODALITIES)
        if model_args.item_table_view == "full"
        else {model_args.item_table_view}
    )
    needed = graph_views | (
        item_views
        if model_args.task == "recommendation"
        and model_args.recommendation_head == "item_table"
        else set()
    )
    directory = Path(data_args.dataset_path) / data_args.embeddings_dir_name
    return {
        view: torch.load(
            directory / f"{view}_embeddings.pt", map_location="cpu", weights_only=True
        ).float()
        for view in MODALITIES
        if view in needed
    }


def _build_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor: HoCRSProcessor,
    tables: dict[str, torch.Tensor],
):
    backbone = load_backbone(model_args.backbone_name_or_path)
    backbone.resize_token_embeddings(len(processor.tokenizer))
    num_items = next(iter(tables.values())).size(0)
    graph_configs = {}
    for view in data_args.views:
        feature_view = model_args.co_feature_view if view == "co" else view
        graph_configs[view] = HoCRSHypergraphConfig(
            input_dim=tables[feature_view].size(1),
            hidden_dim=model_args.hypergraph_hidden_dim,
            output_dim=model_args.hypergraph_output_dim,
            num_layers=model_args.hypergraph_num_layers,
            dropout=model_args.hypergraph_dropout,
        )
    config = HoCRSConfig(
        backbone_config=backbone.config,
        task=model_args.task,
        recommendation_head=model_args.recommendation_head,
        views=data_args.views,
        co_feature_view=model_args.co_feature_view,
        num_items=num_items,
        item_feature_dims={view: table.size(1) for view, table in tables.items()},
        item_table_view=model_args.item_table_view,
        grounding_checkpoint_path=model_args.grounding_checkpoint_path,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_temperature=model_args.recommendation_temperature,
        num_prompt_tokens=model_args.num_prompt_tokens,
        freeze_backbone=model_args.freeze_backbone,
        prompt_token_id=processor.get_token_id_map()["soft_prompt_token_id"],
        **{f"{view}_hypergraph_config": value for view, value in graph_configs.items()},
    )
    model_class = (
        HoCRSRecommendationModel
        if model_args.task == "recommendation"
        else HoCRSConversationModel
    )
    model = model_class(config, backbone=backbone)
    if model_args.grounding_checkpoint_path:
        model.load_grounding_checkpoint(model_args.grounding_checkpoint_path)
    model.initialize_feature_tables(tables)
    return model


class TestEvaluationCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        eval_steps = args.eval_steps
        if eval_steps is None or state.global_step % int(eval_steps) != 0:
            return control
        metrics = self.trainer.evaluate(eval_dataset=self.dataset, metric_key_prefix="test")
        if self.trainer.is_world_process_zero():
            self.trainer.save_metrics("test", metrics)
        return control


def _save_provenance(output_dir, config_path, model_args, data_args, training_args):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"source_experiment_config{config_path.suffix}").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (destination / "resolved_experiment_config.json").write_text(
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


def main() -> None:
    load_dotenv()
    model_args, data_args, training_args, config_path = parse_experiment_args()
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(
        tokenizer=tokenizer,
        num_prompt_tokens=model_args.num_prompt_tokens
    )
    tables = _load_modality_tables(data_args, model_args)
    model = _build_model(model_args, data_args, processor, tables)
    table_path = data_args.hyperedge_table_path or str(Path(data_args.dataset_path) / "hyperedge_table.json")
    hypergraph_table = HypergraphTable.from_json(table_path)
    train_config = HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        task=model_args.task,
        hyperedge_table_path=table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
        sampling=data_args.hyperedge_sampling,
        sample_repeat=data_args.sample_repeat,
    )
    eval_config = HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        task=model_args.task,
        hyperedge_table_path=table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
    )
    train_dataset = (
        HoCRSDataset(train_config, "train", hypergraph_table)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        HoCRSDataset(eval_config, "validation", hypergraph_table)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        HoCRSDataset(eval_config, "test", hypergraph_table)
        if training_args.do_train or training_args.do_predict
        else None
    )
    training_args.label_names = (
        ["rec_labels"] if model_args.task == "recommendation" else ["labels"]
    )
    callback = (
        TestEvaluationCallback(test_dataset) if test_dataset is not None else None
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=HoCRSDataCollator(
            processor,
            model_args.task,
            data_args.max_history_tokens
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=build_compute_metrics(processor, model_args.task),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics(model_args.task),
        callbacks=[callback] if callback else None,
    )
    if callback:
        callback.trainer = trainer
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
        _save_provenance(
            training_args.output_dir, config_path, model_args, data_args, training_args
        )
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        if checkpoint is None and Path(training_args.output_dir).is_dir():
            checkpoint = get_last_checkpoint(training_args.output_dir)
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
    if eval_dataset is not None:
        trainer.save_metrics("eval", trainer.evaluate(metric_key_prefix="eval"))
    if test_dataset is not None:
        trainer.save_metrics("test", trainer.predict(test_dataset, metric_key_prefix="test").metrics)


if __name__ == "__main__":
    main()

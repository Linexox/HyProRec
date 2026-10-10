"""Train the lightweight source and hypergraph towers for Grounding."""

from __future__ import annotations

import json
import copy
import gc
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, set_seed

from ..arguments import DataArguments, ModelArguments
from ..config import parse_experiment_args
from ..configuration_grounding import HoCRSGroundingConfig
from ..constants import MODALITIES
from ..data.grounding import (
    GroundingSourceDataset,
    HoCRSGroundingCollator,
    HoCRSGroundingDataset,
)
from ..data.hypergraph import HypergraphTable
from ..modeling_grounding import HoCRSGroundingModel, build_source_config
from ..tracking import SwanLabTracker


class GroundingTrainer(Trainer):
    def __init__(self, *args, swanlab_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.swanlab_tracker = swanlab_tracker
        self._grounding_sum = {}
        self._grounding_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        for name in ("loss", "ga_sa_loss", "ga_sn_loss", "sa_sn_loss"):
            value = getattr(outputs, name, None)
            if value is not None:
                self._grounding_sum[name] = self._grounding_sum.get(name, 0.0) + float(
                    value.detach().float().cpu()
                )
        self._grounding_count += 1
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._grounding_count:
            for name, value in self._grounding_sum.items():
                logs[name] = value / self._grounding_count
            self._grounding_sum.clear()
            self._grounding_count = 0
        if self.swanlab_tracker is not None and self.is_world_process_zero():
            self.swanlab_tracker.log_grounding(logs)
        return super().log(logs, *args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer


def _load_feature_tables(data_args: DataArguments) -> dict[str, torch.Tensor]:
    embedding_dir = Path(data_args.dataset_path) / data_args.embeddings_dir_name
    return {
        view: torch.load(
            embedding_dir / f"{view}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        for view in MODALITIES
    }


def _finalize_grounding(
    is_main_process,
    output_dir,
    config,
    combined_state,
    selections,
    config_path,
    model_args,
    data_args,
    training_args,
):
    if not is_main_process:
        return
    model = HoCRSGroundingModel(config)
    model.load_state_dict(combined_state, strict=True)
    model.save_pretrained(output_dir)
    output_path = Path(output_dir)
    (output_path / "best_modalities.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"grounding_recipe{config_path.suffix}").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output_path / "resolved_grounding_config.json").write_text(
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
    training_args.report_to = []
    set_seed(training_args.seed)
    tracker = SwanLabTracker.start(model_args, data_args, training_args, config_path)
    views = tuple(data_args.grounding_views or MODALITIES)
    feature_tables = _load_feature_tables(data_args)
    input_dims = {
        view: feature_tables[view].size(1) for view in views
    }
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
    table_path = data_args.hyperedge_table_path or (
        Path(data_args.dataset_path) / "hyperedge_table.json"
    )
    table = HypergraphTable.from_json(table_path)
    combined_state = {}
    selections = {}
    is_main_process = False
    for view in views:
        set_seed(training_args.seed)
        view_config = copy.deepcopy(config)
        view_config.views = (view,)
        tokenizer = None
        source_view = view
        if source_view == "txt":
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.grounding_tokenizer_name_or_path
            )
            source_config = build_source_config(view)
            source_config.vocab_size = len(tokenizer)
            view_config.source_configs[view] = source_config.to_dict()
        source_dataset = GroundingSourceDataset(data_args.dataset_path, (source_view,))
        if not source_dataset.has_raw_modality(source_view):
            view_config.source_configs[view] = {
                "model_type": "feature_projection",
                "input_dim": input_dims[view],
            }
        model = HoCRSGroundingModel(view_config)
        config.source_configs.update(view_config.source_configs)
        args = copy.deepcopy(training_args)
        args.output_dir = str(Path(training_args.output_dir) / view)
        args.run_name = view.replace("_", "-")
        args.label_names = []
        args.prediction_loss_only = True
        args.eval_strategy = "epoch"
        args.save_strategy = "epoch"
        args.metric_for_best_model = "eval_loss"
        args.greater_is_better = False
        dataset = HoCRSGroundingDataset(
            table, (view,), topk=data_args.topk, khop=data_args.khop
        )
        split = int(len(dataset) * 0.9)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset,
            [split, len(dataset) - split],
            generator=torch.Generator().manual_seed(training_args.seed),
        )
        trainer = GroundingTrainer(
            model=model,
            args=args,
            data_collator=HoCRSGroundingCollator(
                {source_view: feature_tables[source_view]},
                (view,),
                source_dataset,
                tokenizer,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            swanlab_tracker=tracker,
        )
        result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()  # Trainer has restored this modality's minimum-eval-loss model.
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        if trainer.is_world_process_zero():
            combined_state.update(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )
            selections[view] = {
                "checkpoint": trainer.state.best_model_checkpoint,
                "eval_loss": trainer.state.best_metric,
            }

        trainer.accelerator.wait_for_everyone()
        is_main_process = trainer.is_world_process_zero()
        trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _finalize_grounding(
        is_main_process,
        training_args.output_dir,
        config,
        combined_state,
        selections,
        config_path,
        model_args,
        data_args,
        training_args,
    )
    tracker.finish()


if __name__ == "__main__":
    main()

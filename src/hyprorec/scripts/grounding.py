"""Train the lightweight source and hypergraph towers for Grounding."""

from __future__ import annotations

import json
import copy
import gc
from dataclasses import asdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, Trainer, set_seed

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


# START: HoCRS2 applies AdamW weight decay to every trainable parameter.
class GroundingTrainer(Trainer):
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


# END: HoCRS2 applies AdamW weight decay to every trainable parameter.


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
    set_seed(training_args.seed)  # Modified: seed before any module construction.
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
    table_path = data_args.hyperedge_table_path or (
        Path(data_args.dataset_path) / "hyperedge_table.json"
    )
    table = HypergraphTable.from_json(table_path)
    # START: Each modality gets its own optimizer and validation-selected best model.
    if (
        not training_args.do_train
        or not training_args.do_eval
        or not training_args.load_best_model_at_end
    ):
        raise ValueError(
            "Grounding requires training, validation, and load_best_model_at_end."
        )
    combined_state = {}
    selections = {}
    for view in views:
        set_seed(training_args.seed)
        view_config = copy.deepcopy(config)
        view_config.views = (view,)
        tokenizer = None
        if view == "txt":
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.grounding_tokenizer_name_or_path
            )
            source_config = build_source_config(view)
            source_config.vocab_size = len(tokenizer)
            view_config.source_configs[view] = source_config.to_dict()
        model = HoCRSGroundingModel(view_config)
        config.source_configs.update(view_config.source_configs)
        args = copy.deepcopy(training_args)
        args.output_dir = str(Path(training_args.output_dir) / view)
        args.run_name = f"{training_args.run_name or 'grounding'}-{view}"
        args.label_names = []
        args.prediction_loss_only = True
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
                {view: feature_tables[view]},
                (view,),
                GroundingSourceDataset(data_args.dataset_path, (view,)),
                tokenizer,
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
            if "wandb" in args.report_to:
                import wandb

                wandb.finish()
        trainer.accelerator.wait_for_everyone()
        is_main_process = trainer.is_world_process_zero()
        trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if is_main_process:
        model = HoCRSGroundingModel(config)
        model.load_state_dict(combined_state, strict=True)
        model.save_pretrained(training_args.output_dir)
        (Path(training_args.output_dir) / "best_modalities.json").write_text(
            json.dumps(selections, indent=2), encoding="utf-8"
        )
        training_args.label_names = []
        training_args.prediction_loss_only = True
    # END: Each modality gets its own optimizer and validation-selected best model.
    if is_main_process:
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


if __name__ == "__main__":
    main()

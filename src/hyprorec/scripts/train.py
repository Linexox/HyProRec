"""Train HyProRec with the native Transformers training and evaluation loops."""

from __future__ import annotations

import math
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from ..config import parse_experiment_args
from ..data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from ..data.hypergraph import HypergraphTable
from ..evaluation import build_compute_metrics
from ..model_factory import build_model, load_modality_tables
from ..processing_hocrs import HoCRSProcessor
from ..provenance import save_provenance
from ..training import HyProRecTrainer


def _dataset_config(data_args, table_path, *, training):
    return HoCRSDatasetConfig(
        dataset_path=data_args.dataset_path,
        hyperedge_table_path=table_path,
        views=tuple(data_args.views),
        topk=data_args.topk,
        khop=data_args.khop,
        sampling=data_args.hyperedge_sampling if training else "strict",
        sample_repeat=data_args.sample_repeat if training else 1,
        max_hypergraph_nodes=data_args.max_hypergraph_nodes,
    )


def _configure_step_schedule(trainer, training_args):
    """Use the configured step-based Trainer loop with four checks per epoch."""

    batches = len(trainer.get_train_dataloader())
    updates = max(1, math.ceil(batches / max(1, training_args.gradient_accumulation_steps)))
    interval = max(1, math.ceil(updates / 4))
    training_args.eval_steps = interval
    training_args.save_steps = interval
    training_args.save_total_limit = 2
    training_args.load_best_model_at_end = True
    # Keep the metric requested by the experiment config.  In particular,
    # lambda sweeps select checkpoints by validation Recall@50 rather than the
    # lambda-weighted joint loss, which is not comparable across runs.
    if not training_args.metric_for_best_model:
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False


def main() -> None:
    load_dotenv()
    model_args, data_args, training_args, config_path = parse_experiment_args()
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(tokenizer, model_args.num_prompt_tokens)
    tables = load_modality_tables(data_args, model_args)
    model = build_model(model_args, data_args, processor, tables)

    table_path = data_args.hyperedge_table_path or str(
        Path(data_args.dataset_path) / "hyperedge_table.json"
    )
    hypergraph_table = HypergraphTable.from_json(table_path)
    train_dataset = (
        HoCRSDataset(
            _dataset_config(data_args, table_path, training=True),
            "train",
            hypergraph_table,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        HoCRSDataset(
            _dataset_config(data_args, table_path, training=False),
            "validation",
            hypergraph_table,
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        HoCRSDataset(
            _dataset_config(data_args, table_path, training=False),
            "test",
            hypergraph_table,
        )
        if training_args.do_predict
        else None
    )

    training_args.label_names = ["labels", "rec_labels"]
    trainer = HyProRecTrainer(
        model=model,
        args=training_args,
        data_collator=HoCRSDataCollator(
            processor,
            data_args.max_history_tokens,
            data_args.max_sequence_tokens,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=build_compute_metrics(tokenizer),
        attention_token_ids=processor.get_token_id_map(),
    )
    if training_args.do_train and train_dataset is not None:
        _configure_step_schedule(trainer, training_args)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)
        save_provenance(
            training_args.output_dir,
            config_path,
            model_args,
            data_args,
            training_args,
        )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        if checkpoint is None and Path(training_args.output_dir).is_dir():
            checkpoint = get_last_checkpoint(training_args.output_dir)
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)

    if eval_dataset is not None:
        valid_metrics = trainer.evaluate(
            eval_dataset=eval_dataset, metric_key_prefix="eval"
        )
        trainer.save_metrics("valid", valid_metrics)

    if test_dataset is not None:
        test_metrics = trainer.predict(
            test_dataset, metric_key_prefix="test"
        ).metrics
        trainer.log(test_metrics)
        trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()

"""Pretrain source encoders in a common multimodal item space."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import Subset
from transformers import Trainer

from ..config import parse_alignment_args
from ..configuration_alignment import HoCRSAlignmentConfig
from ..data.alignment import (
    HoCRSAlignmentCollator,
    HoCRSAlignmentDataset,
    split_item_ids,
)
from ..modeling_alignment import HoCRSAlignmentModel


def main() -> None:
    load_dotenv()
    alignment_args, training_args, config_path = parse_alignment_args()
    config = HoCRSAlignmentConfig(
        modalities=alignment_args.modalities,
        txt_model_name_or_path=alignment_args.txt_model_name_or_path,
        img_model_name_or_path=alignment_args.img_model_name_or_path,
        ado_model_name_or_path=alignment_args.ado_model_name_or_path,
        vdo_model_name_or_path=alignment_args.vdo_model_name_or_path,
        alignment_dim=alignment_args.alignment_dim,
        temperature=alignment_args.temperature,
    )
    model = HoCRSAlignmentModel(config)
    dataset = HoCRSAlignmentDataset(
        alignment_args.dataset_path, alignment_args.modalities
    )
    train_ids, validation_ids = split_item_ids(
        len(dataset), alignment_args.validation_ratio, training_args.data_seed
    )
    train_dataset = Subset(dataset, train_ids)
    eval_dataset = Subset(dataset, validation_ids)
    collator = HoCRSAlignmentCollator(config, alignment_args.max_text_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
    )

    if trainer.is_world_process_zero():
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"alignment_recipe{config_path.suffix}").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (output_dir / "resolved_alignment_config.json").write_text(
            json.dumps(
                {
                    "alignment": asdict(alignment_args),
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

"""Persist the resolved configuration used by an experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def save_provenance(output_dir, config_path, model_args, data_args, training_args):
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


__all__ = ["save_provenance"]

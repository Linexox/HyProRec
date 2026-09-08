"""Sectioned YAML/JSON experiment parsing with normal CLI overrides."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import yaml
from transformers import HfArgumentParser

from .arguments import (
    AlignmentArguments,
    DataArguments,
    HoCRSTrainingArguments,
    ModelArguments,
)

CONFIG_SECTIONS = (
    ("model", ModelArguments),
    ("data", DataArguments),
    ("training", HoCRSTrainingArguments),
)


def parse_experiment_args(
    args: Sequence[str] | None = None,
) -> tuple[ModelArguments, DataArguments, HoCRSTrainingArguments, Path]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, required=True)
    config_namespace, _ = config_parser.parse_known_args(args)
    config_path = config_namespace.config.resolve()
    with config_path.open(encoding="utf-8") as file:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(file)
        elif config_path.suffix.lower() == ".json":
            payload = json.load(file)
        else:
            raise ValueError("Experiment configs must be YAML or JSON files.")
    if not isinstance(payload, dict):
        raise TypeError("The experiment config root must be an object.")

    expected_sections = {name for name, _ in CONFIG_SECTIONS}
    unknown_sections = set(payload) - expected_sections
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    defaults: dict[str, Any] = {}
    for section_name, dataclass_type in CONFIG_SECTIONS:
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(f"Config section '{section_name}' must be an object.")
        valid_fields = {field.name for field in fields(dataclass_type) if field.init}
        unknown_fields = set(section) - valid_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown fields in '{section_name}': {sorted(unknown_fields)}"
            )
        overlap = set(defaults) & set(section)
        if overlap:
            raise ValueError(f"Config fields have multiple owners: {sorted(overlap)}")
        defaults.update(section)

    parser = HfArgumentParser(
        tuple(dataclass_type for _, dataclass_type in CONFIG_SECTIONS)
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.set_defaults(**defaults)
    model_args, data_args, training_args, remaining = (
        parser.parse_args_into_dataclasses(args=args)
    )
    return model_args, data_args, training_args, remaining.config.resolve()


# START: Parse alignment recipes with the same YAML plus CLI override convention.
def parse_alignment_args(
    args: Sequence[str] | None = None,
) -> tuple[AlignmentArguments, HoCRSTrainingArguments, Path]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, required=True)
    config_namespace, _ = config_parser.parse_known_args(args)
    config_path = config_namespace.config.resolve()
    with config_path.open(encoding="utf-8") as file:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(file)
        elif config_path.suffix.lower() == ".json":
            payload = json.load(file)
        else:
            raise ValueError("Alignment configs must be YAML or JSON files.")
    if not isinstance(payload, dict):
        raise TypeError("The alignment config root must be an object.")
    expected = {"alignment", "training"}
    unknown_sections = set(payload) - expected
    if unknown_sections:
        raise ValueError(
            f"Unknown alignment config sections: {sorted(unknown_sections)}"
        )

    defaults: dict[str, Any] = {}
    for section_name, dataclass_type in (
        ("alignment", AlignmentArguments),
        ("training", HoCRSTrainingArguments),
    ):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(f"Config section '{section_name}' must be an object.")
        valid_fields = {field.name for field in fields(dataclass_type) if field.init}
        unknown_fields = set(section) - valid_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown fields in '{section_name}': {sorted(unknown_fields)}"
            )
        defaults.update(section)

    parser = HfArgumentParser((AlignmentArguments, HoCRSTrainingArguments))
    parser.add_argument("--config", type=Path, required=True)
    parser.set_defaults(**defaults)
    alignment_args, training_args, remaining = parser.parse_args_into_dataclasses(
        args=args
    )
    return alignment_args, training_args, remaining.config.resolve()


# END: Parse alignment recipes with the same YAML plus CLI override convention.


__all__ = ["CONFIG_SECTIONS", "parse_alignment_args", "parse_experiment_args"]

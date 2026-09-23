"""Combine reused semantic towers with newly grounded co-occurrence towers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

CO_VIEWS = ("co_txt", "co_img", "co_vdo", "co_ado")
ENCODER_PREFIX = "hypergraph_encoders."


def merge_checkpoints(base_dir: Path, co_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    co_config = json.loads((co_dir / "config.json").read_text(encoding="utf-8"))
    expected_views = set(base_config["views"])
    if not set(CO_VIEWS).issubset(expected_views):
        raise ValueError("Base checkpoint does not contain all four co towers.")
    if set(co_config["views"]) != set(CO_VIEWS):
        raise ValueError("Co checkpoint must contain exactly co_txt/img/vdo/ado.")

    base_state = load_file(str(base_dir / "model.safetensors"), device="cpu")
    co_state = load_file(str(co_dir / "model.safetensors"), device="cpu")
    merged_state = dict(base_state)
    for view in CO_VIEWS:
        prefix = f"{ENCODER_PREFIX}{view}."
        base_keys = {key for key in base_state if key.startswith(prefix)}
        co_keys = {key for key in co_state if key.startswith(prefix)}
        if not base_keys or base_keys != co_keys:
            raise ValueError(f"Checkpoint keys do not match for {view}.")
        merged_state.update({key: co_state[key] for key in co_keys})

    output_dir.mkdir(parents=True)
    save_file(merged_state, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(
        json.dumps(base_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selections = json.loads(
        (base_dir / "best_modalities.json").read_text(encoding="utf-8")
    )
    co_selections = json.loads(
        (co_dir / "best_modalities.json").read_text(encoding="utf-8")
    )
    selections.update({view: co_selections[view] for view in CO_VIEWS})
    (output_dir / "best_modalities.json").write_text(
        json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "merge_provenance.json").write_text(
        json.dumps(
            {
                "semantic_towers_from": str(base_dir),
                "co_towers_from": str(co_dir),
                "replaced_views": list(CO_VIEWS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--co", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge_checkpoints(args.base, args.co, args.output)


if __name__ == "__main__":
    main()

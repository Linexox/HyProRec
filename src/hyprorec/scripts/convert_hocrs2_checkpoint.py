"""Convert a HoCRS2 delta checkpoint to a HyProRec checkpoint."""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

from ..configuration_hocrs import HoCRSHypergraphConfig
from ..modeling_hocrs import HypergraphEncoder, HypergraphProjector

VIEWS = ("txt", "img", "ado", "vdo")


def convert_checkpoint(source: Path, output: Path) -> None:
    # START: Convert only known Grounding modules and preserve full provenance.
    source = source.resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("adapter_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("HoCRS2 checkpoint must contain adapter_state_dict.")
    converted: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    mapping = {}
    config_path = source.parent / "config.json"
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    for view in VIEWS:
        tower_prefix = f"{view}_hypergraph_tower."
        projector_prefix = f"{view}_hypergraph_projector."
        tower = {
            f"hypergraph_encoders.{view}.{name.removeprefix(tower_prefix)}": value
            for name, value in state.items()
            if name.startswith(tower_prefix)
        }
        projector = {
            f"hypergraph_projectors.{view}.{name.removeprefix(projector_prefix)}": value
            for name, value in state.items()
            if name.startswith(projector_prefix)
        }
        if len(tower) != 12 or len(projector) != 4:
            raise ValueError(
                f"{view}: expected 12 tower and 4 projector tensors, "
                f"got {len(tower)} and {len(projector)}."
            )
        graph_config = HoCRSHypergraphConfig(**{
            key: source_config[f"{view}_hypergraph_config"][key]
            for key in ("input_dim", "hidden_dim", "output_dim", "num_layers", "dropout", "use_bias")
        })
        HypergraphEncoder(graph_config).load_state_dict({
            key.removeprefix(f"hypergraph_encoders.{view}."): value for key, value in tower.items()
        }, strict=True)
        HypergraphProjector(graph_config.output_dim, 2048).load_state_dict({
            key.removeprefix(f"hypergraph_projectors.{view}."): value for key, value in projector.items()
        }, strict=True)
        mapping.update({key: key.replace(f"hypergraph_encoders.{view}.", tower_prefix)
                        for key in tower})
        mapping.update({key: key.replace(f"hypergraph_projectors.{view}.", projector_prefix)
                        for key in projector})
        converted.update(tower)
        converted.update(projector)
        counts[view] = len(tower) + len(projector)
    output.mkdir(parents=True, exist_ok=False)
    save_file(converted, str(output / "model.safetensors"))
    saved = load_file(str(output / "model.safetensors"))
    for target, origin in mapping.items():
        torch.testing.assert_close(saved[target], state[origin], rtol=0, atol=0)
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "hocrs2_source_config.json").write_text(json.dumps(source_config, indent=2), encoding="utf-8")
    (output / "conversion_log.json").write_text(
        json.dumps({"source": str(source), "source_sha256": sha256(source),
                    "source_config_sha256": sha256(config_path),
                    "output_sha256": sha256(output / "model.safetensors"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "views": counts, "tensor_count": len(converted), "mapping": mapping,
                    "purpose": "CRS initialization: frozen towers and trainable projectors; no TASK or source encoder weights",
                    "tensor_equality_verified": True}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "tensor_count": len(converted)}, indent=2))
    # END: Convert only known Grounding modules and preserve full provenance.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()

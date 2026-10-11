"""SwanLab experiment tracking for reproducible HyProRec runs."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            result[name] = list(value)
    return result


class SwanLabTracker:
    """Small adapter so the trainer stays independent of the tracking SDK."""

    def __init__(self, run=None) -> None:
        self.run = run

    @classmethod
    def start(cls, model_args, data_args, training_args, config_path: Path):
        if os.environ.get("RANK", "0") != "0":
            return cls()
        mode = os.getenv("SWANLAB_MODE", "online")
        if mode == "disabled":
            return cls()
        import swanlab

        api_key = os.getenv("SWANLAB_API_KEY")
        api_host = os.getenv("SWANLAB_API_HOST")
        if api_key or api_host:
            swanlab.login(api_key=api_key, host=api_host, save=False)

        dataset = Path(data_args.dataset_path).name
        backbone = Path(model_args.backbone_name_or_path).name
        views = "-".join(data_args.views)
        composition = (
            "full"
            if tuple(data_args.views) == ("txt", "img", "ado", "vdo")
            else f"only-{views}"
        )
        requested_name = training_args.run_name
        name = (
            requested_name
            if requested_name and requested_name.startswith(("only-", "full"))
            else composition
        )
        tags = [
            f"dataset:{dataset}",
            f"views:{views}",
            f"backbone:{backbone}",
            f"hidden:{model_args.recommendation_hidden_dim}",
            f"nodes:{data_args.max_hypergraph_nodes}",
            f"lambda:{model_args.conversation_loss_weight}",
            f"eta:{model_args.recommendation_view_loss_weight}",
            f"seed:{training_args.seed}",
        ]
        config = _flatten(
            {
                "model": asdict(model_args),
                "data": asdict(data_args),
                "training": training_args.to_dict(),
                "config_path": str(config_path),
            }
        )
        run = swanlab.init(
            project=os.getenv("SWANLAB_PROJECT", "HyProRec"),
            name=name,
            tags=tags,
            config=config,
            mode=mode,
        )
        return cls(run)

    def log(self, metrics: dict[str, Any]) -> None:
        if self.run is None:
            return
        import swanlab

        values = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value == value
        }
        if values:
            swanlab.log(values)

    def log_trainer(self, logs: dict[str, Any]) -> None:
        """Map Transformers' log names to the stable HyProRec panels."""
        if self.run is None:
            return
        stage = "train"
        if any(key.startswith("test_") for key in logs):
            stage = "test"
        elif any(key.startswith("eval_") for key in logs):
            stage = "valid"
        main: dict[str, Any] = {}
        other: dict[str, Any] = {}
        for key, value in logs.items():
            clean = key.removeprefix("eval_").removeprefix("test_")
            if clean.startswith("attention/"):
                _, view, metric = clean.split("/", 2)
                allowed = {
                    "soft_prompt_mass": f"{view}-soft-pro-mass",
                    "context_mass": f"{view}-context-mass",
                }
                if metric in allowed:
                    other[f"other_{stage}/AttnPooling/{allowed[metric]}"] = value
                elif metric.endswith("_hypergraph_mass"):
                    source = metric.removesuffix("_hypergraph_mass")
                    other[
                        f"other_{stage}/AttnPooling/{view}-{source}-hyg-mass"
                    ] = value
            elif clean.startswith("rec_router/mean_"):
                view = clean.removeprefix("rec_router/mean_")
                other[f"other_{stage}/RecRouter/{view}"] = value
            elif clean.startswith(
                (
                    "loss/",
                    "recall@",
                    "ndcg@",
                    "mrr@",
                    "bleu@",
                    "dist@",
                    "response_tokens",
                )
            ):
                main[f"{stage}/{clean}"] = value
        self.log({**main, **other})

    def log_grounding(self, logs: dict[str, Any]) -> None:
        stage = "valid" if any(key.startswith("eval_") for key in logs) else "train"
        values = {}
        names = {
            "loss": "total",
            "ga_sa_loss": "ga-sa",
            "ga_sn_loss": "ga-sn",
            "sa_sn_loss": "sa-sn",
        }
        for key, value in logs.items():
            clean = key.removeprefix("eval_")
            if clean in names:
                values[f"{stage}/grounding/{names[clean]}"] = value
        self.log(values)

    def finish(self, state: str = "success", error: str | None = None) -> None:
        if self.run is None:
            return
        import swanlab

        kwargs = {"state": state}
        if error:
            kwargs["error"] = error
        swanlab.finish(**kwargs)


__all__ = ["SwanLabTracker"]

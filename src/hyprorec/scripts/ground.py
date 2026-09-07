"""Pre-train one Grounding graph tower per configured view."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import parse_grounding_args
from ..data.grounding import HoCRSGroundingDataset
from ..grounding import HoCRSGroundingModel


def _run_view(config, view: str, device: torch.device, output_dir: Path) -> None:
    view_config = replace(config, modalities=[view])
    train_dataset = HoCRSGroundingDataset(view_config, "train")
    valid_dataset = HoCRSGroundingDataset(view_config, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn,
    )
    input_dim = train_dataset.graph_features[view].size(1)
    model = HoCRSGroundingModel(view_config, view, int(input_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                loss, _ = model(batch)
                valid_total += float(loss)
        train_mean = train_total / max(1, len(train_loader))
        valid_mean = valid_total / max(1, len(valid_loader))
        print(f"{view} epoch {epoch}: train={train_mean:.4f} valid={valid_mean:.4f}")
        if valid_mean < best_loss:
            best_loss = valid_mean
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"No Grounding checkpoint was produced for {view}.")
    destination = output_dir / view
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, destination / "pytorch_model.bin")
    (destination / "grounding_config.json").write_text(
        json.dumps(
            {
                "view": view,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                **asdict(view_config),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    config, config_path = parse_grounding_args()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"grounding_recipe{config_path.suffix}").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for view in config.modalities:
        _run_view(config, view, device, output_dir)


if __name__ == "__main__":
    main()

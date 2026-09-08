"""CLIP-style multimodal alignment for offline item representations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch import nn
from transformers import (
    MPNetModel,
    PreTrainedModel,
    VideoMAEModel,
    ViTModel,
    Wav2Vec2Model,
)
from transformers.modeling_outputs import ModelOutput

from .configuration_alignment import HoCRSAlignmentConfig
from .losses import multi_positive_contrastive_loss


def _gather_alignment_batch(
    embeddings: dict[str, torch.Tensor],
    item_ids: torch.Tensor,
    modality_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Use the effective distributed batch as CLIP-style negatives."""

    if not dist.is_available() or not dist.is_initialized():
        return embeddings, item_ids, modality_mask
    return (
        {
            modality: torch.cat(all_gather(values), dim=0)
            for modality, values in embeddings.items()
        },
        torch.cat(all_gather(item_ids), dim=0),
        torch.cat(all_gather(modality_mask), dim=0),
    )


def _load_video_model(name_or_path: str) -> VideoMAEModel:
    """Load renamed VideoMAE biases without silently dropping pretrained values."""

    model, loading_info = VideoMAEModel.from_pretrained(
        name_or_path,
        key_mapping={r"\.q_bias$": ".query.bias", r"\.v_bias$": ".value.bias"},
        output_loading_info=True,
    )
    missing = set(loading_info.get("missing_keys", []))
    key_biases = {
        f"encoder.layer.{index}.attention.attention.key.bias"
        for index in range(model.config.num_hidden_layers)
    }
    if missing - key_biases:
        raise ValueError(f"VideoMAE weights were not fully loaded: {sorted(missing)}")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in missing:
                parameter.zero_()
    return model


def _hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Cannot determine source encoder hidden size.")


@dataclass
class HoCRSAlignmentOutput(ModelOutput):
    loss: torch.Tensor | None = None
    embeddings: dict[str, torch.Tensor] | None = None


class HoCRSAlignmentModel(PreTrainedModel):
    """Train source encoders into one normalized item representation space."""

    config_class = HoCRSAlignmentConfig
    base_model_prefix = "source_encoders"
    supports_gradient_checkpointing = True

    # START: Composite source encoders must load outside the outer meta context.
    @classmethod
    def get_init_context(cls, *args, **kwargs):
        contexts = super().get_init_context(*args, **kwargs)
        return [
            context
            for context in contexts
            if not (isinstance(context, torch.device) and context.type == "meta")
        ]

    # END: Composite source encoders must load outside the outer meta context.

    def __init__(self, config: HoCRSAlignmentConfig) -> None:
        super().__init__(config)
        loaders = {
            "txt": MPNetModel.from_pretrained,
            "img": ViTModel.from_pretrained,
            "ado": Wav2Vec2Model.from_pretrained,
            "vdo": _load_video_model,
        }
        self.source_encoders = nn.ModuleDict(
            {
                modality: loaders[modality](config.get_model_name_or_path(modality))
                for modality in config.modalities
            }
        )
        self.alignment_heads = nn.ModuleDict(
            {
                modality: nn.Linear(
                    _hidden_size(self.source_encoders[modality].config),
                    config.alignment_dim,
                )
                for modality in config.modalities
            }
        )
        self._no_split_modules = list(
            dict.fromkeys(
                module_name
                for encoder in self.source_encoders.values()
                for module_name in getattr(encoder, "_no_split_modules", [])
            )
        )
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        for encoder in self.source_encoders.values():
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self) -> None:
        for encoder in self.source_encoders.values():
            encoder.gradient_checkpointing_disable()

    def _encode_modality(
        self,
        modality: str,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        output = self.source_encoders[modality](**inputs).last_hidden_state
        if modality == "txt":
            mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        elif modality == "img":
            pooled = output[:, 0]
        else:
            pooled = output.mean(1)
        return F.normalize(self.alignment_heads[modality](pooled), dim=-1)

    def encode(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = {}
        for modality in self.config.modalities:
            prefix = f"{modality}_"
            modality_inputs = {
                name.removeprefix(prefix): value
                for name, value in inputs.items()
                if name.startswith(prefix)
            }
            embeddings[modality] = self._encode_modality(modality, modality_inputs)
        return embeddings

    def forward(
        self,
        item_ids: torch.Tensor,
        modality_mask: torch.Tensor,
        **inputs: torch.Tensor,
    ) -> HoCRSAlignmentOutput:
        embeddings = self.encode(**inputs)
        gathered, gathered_ids, gathered_mask = _gather_alignment_batch(
            embeddings, item_ids, modality_mask
        )
        losses = []
        for left_modality, right_modality in combinations(self.config.modalities, 2):
            left_index = self.config.modalities.index(left_modality)
            right_index = self.config.modalities.index(right_modality)
            valid = gathered_mask[:, left_index] & gathered_mask[:, right_index]
            pair_loss = multi_positive_contrastive_loss(
                gathered[left_modality][valid],
                gathered[right_modality][valid],
                gathered_ids[valid],
                self.config.temperature,
            )
            if pair_loss is not None:
                losses.append(pair_loss)
        if not losses:
            loss = sum(embedding.sum() for embedding in embeddings.values()) * 0.0
        else:
            loss = torch.stack(losses).mean()
        return HoCRSAlignmentOutput(loss=loss, embeddings=embeddings)


__all__ = [
    "HoCRSAlignmentModel",
    "HoCRSAlignmentOutput",
]

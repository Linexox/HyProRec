"""Serializable configuration for the HoCRS Grounding stage."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from .constants import MODALITIES


class HoCRSGroundingConfig(PretrainedConfig):
    model_type = "hocrs_grounding"

    def __init__(
        self,
        views: list[str] | tuple[str, ...] = MODALITIES,
        input_dims: dict[str, int] | None = None,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        temperature: float = 0.07,
        lambda_ga_sa: float = 1.0,
        lambda_ga_sn: float = 3.0,
        lambda_sa_sn: float = 3.0,
        source_configs: dict[str, dict[str, Any]] | None = None,
        num_items: int = 6924,
        id_embedding_dim: int = 768,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.views = tuple(dict.fromkeys(views))
        if not set(self.views).issubset(
            (*MODALITIES, "co", *(f"co_{view}" for view in MODALITIES))
        ):
            raise ValueError(
                "Grounding views must be semantic modalities or modality-initialized co views."
            )
        self.input_dims = dict(input_dims or {})
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.lambda_ga_sa = lambda_ga_sa
        self.lambda_ga_sn = lambda_ga_sn
        self.lambda_sa_sn = lambda_sa_sn
        self.source_configs = dict(source_configs or {})
        self.num_items = int(num_items)
        self.id_embedding_dim = int(id_embedding_dim)


__all__ = ["HoCRSGroundingConfig"]

"""Configuration classes for HoCRS."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, GPT2Config, PretrainedConfig

from .constants import GRAPH_VIEWS


class HoCRSHypergraphConfig(PretrainedConfig):
    """Shape and regularization settings for one hypergraph encoder."""

    model_type = "hocrs_hypergraph"

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bias = use_bias


def _backbone_config(
    value: PretrainedConfig | dict[str, Any] | None,
) -> PretrainedConfig:
    if value is None:
        return GPT2Config()
    if isinstance(value, PretrainedConfig):
        return value
    config_dict = dict(value)
    model_type = config_dict.pop("model_type")
    return AutoConfig.for_model(model_type, **config_dict)


def _hypergraph_config(
    value: HoCRSHypergraphConfig | dict[str, Any] | None,
) -> HoCRSHypergraphConfig:
    if value is None:
        return HoCRSHypergraphConfig()
    if isinstance(value, HoCRSHypergraphConfig):
        return value
    return HoCRSHypergraphConfig(**value)


class HoCRSConfig(PretrainedConfig):
    """Complete serializable configuration for the first HyProRec model."""

    model_type = "hocrs"
    is_composition = True
    keys_to_ignore_at_inference = [
        "past_key_values",
        "hidden_states",
        "attentions",
    ]

    def __init__(
        self,
        backbone_config: PretrainedConfig | dict[str, Any] | None = None,
        views: list[str] | tuple[str, ...] = GRAPH_VIEWS,
        co_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        txt_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        img_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        ado_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        vdo_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        num_items: int = 6924,
        item_dim: int = 768,
        use_hypergraph_encoder: bool = True,
        recommendation_hidden_dim: int = 768,
        recommendation_dropout: float = 0.0,
        recommendation_temperature: float = 0.07,
        beta: float = 0.75,
        num_soft_prompt_tokens: int = 10,
        freeze_backbone: bool = True,
        train_special_tokens: bool = True,
        node_token_id: int | None = None,
        hyperedge_token_id: int | None = None,
        rec_token_id: int | None = None,
        soft_prompt_token_id: int | None = None,
        graph_start_token_ids: dict[str, int] | None = None,
        graph_end_token_ids: dict[str, int] | None = None,
        trainable_special_token_ids: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        backbone_config = _backbone_config(backbone_config)
        self.backbone_config = backbone_config
        for token_name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            kwargs.setdefault(token_name, getattr(backbone_config, token_name, None))
        super().__init__(**kwargs)
        views = tuple(dict.fromkeys(views))
        unknown_views = set(views) - set(GRAPH_VIEWS)
        if unknown_views:
            raise ValueError(f"Unknown graph views: {sorted(unknown_views)}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        if num_items <= 0:
            raise ValueError("num_items must be positive.")

        self.views = views
        self.co_hypergraph_config = _hypergraph_config(co_hypergraph_config)
        self.txt_hypergraph_config = _hypergraph_config(txt_hypergraph_config)
        self.img_hypergraph_config = _hypergraph_config(img_hypergraph_config)
        self.ado_hypergraph_config = _hypergraph_config(ado_hypergraph_config)
        self.vdo_hypergraph_config = _hypergraph_config(vdo_hypergraph_config)
        self.num_items = num_items
        self.item_dim = item_dim
        self.use_hypergraph_encoder = use_hypergraph_encoder
        self.recommendation_hidden_dim = recommendation_hidden_dim
        self.recommendation_dropout = recommendation_dropout
        self.recommendation_temperature = recommendation_temperature
        self.beta = beta
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
        self.freeze_backbone = freeze_backbone
        self.train_special_tokens = train_special_tokens
        self.node_token_id = node_token_id
        self.hyperedge_token_id = hyperedge_token_id
        self.rec_token_id = rec_token_id
        self.soft_prompt_token_id = soft_prompt_token_id
        self.graph_start_token_ids = graph_start_token_ids or {}
        self.graph_end_token_ids = graph_end_token_ids or {}
        self.trainable_special_token_ids = tuple(trainable_special_token_ids or ())

    def get_hypergraph_config(self, view: str) -> HoCRSHypergraphConfig:
        if view not in GRAPH_VIEWS:
            raise KeyError(f"Unknown graph view: {view}")
        return getattr(self, f"{view}_hypergraph_config")

    def get_text_config(
        self,
        decoder: bool | None = None,
        encoder: bool | None = None,
    ) -> PretrainedConfig:
        del decoder, encoder
        return self.backbone_config


__all__ = ["HoCRSConfig", "HoCRSHypergraphConfig"]

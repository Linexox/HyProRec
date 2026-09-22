"""Configuration for the decoupled HyProRec models."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, GPT2Config, PretrainedConfig

from .constants import GRAPH_VIEWS, MODALITIES

ALL_GRAPH_VIEWS = (*GRAPH_VIEWS, "co")


class HoCRSHypergraphConfig(PretrainedConfig):
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
        if min(input_dim, hidden_dim, output_dim, num_layers) < 1:
            raise ValueError("Hypergraph dimensions and num_layers must be positive.")
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
    values = dict(value)
    model_type = values.pop("model_type")
    return AutoConfig.for_model(model_type, **values)


def _graph_config(
    value: HoCRSHypergraphConfig | dict[str, Any] | None,
) -> HoCRSHypergraphConfig:
    if value is None:
        return HoCRSHypergraphConfig()
    return (
        value
        if isinstance(value, HoCRSHypergraphConfig)
        else HoCRSHypergraphConfig(**value)
    )


class HoCRSConfig(PretrainedConfig):
    model_type = "hocrs"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values", "hidden_states", "attentions"]

    def __init__(
        self,
        backbone_config: PretrainedConfig | dict[str, Any] | None = None,
        task: str = "recommendation",
        recommendation_head: str = "item_table",
        views: list[str] | tuple[str, ...] = ALL_GRAPH_VIEWS,
        co_feature_view: str = "txt",
        txt_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        img_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        ado_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        vdo_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        co_hypergraph_config: HoCRSHypergraphConfig | dict[str, Any] | None = None,
        num_items: int = 6924,
        item_feature_dims: dict[str, int] | None = None,
        item_table_view: str = "txt",
        grounding_checkpoint_path: str | None = None,
        recommendation_hidden_dim: int = 2048,
        recommendation_temperature: float = 0.07,
        num_prompt_tokens: int = 20,
        freeze_backbone: bool = True,
        prompt_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        backbone_config = _backbone_config(backbone_config)
        self.backbone_config = backbone_config
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            kwargs.setdefault(name, getattr(backbone_config, name, None))
        super().__init__(**kwargs)
        views = tuple(dict.fromkeys(views))
        if task not in {"recommendation", "conversation"}:
            raise ValueError("task must be recommendation or conversation.")
        if recommendation_head not in {"item_table", "mlp"}:
            raise ValueError("recommendation_head must be item_table or mlp.")
        if set(views) - set(ALL_GRAPH_VIEWS):
            raise ValueError(f"views must contain each graph view at most once: {ALL_GRAPH_VIEWS}")
        if co_feature_view not in MODALITIES:
            raise ValueError(f"co_feature_view must be one of {MODALITIES}.")
        if item_table_view not in (*MODALITIES, "full"):
            raise ValueError("item_table_view must be a modality or full.")
        if num_items < 1 or recommendation_hidden_dim < 1 or num_prompt_tokens != 20:
            raise ValueError("num_items and recommendation_hidden_dim must be positive; prompts must have length 20.")
        if recommendation_temperature <= 0:
            raise ValueError("recommendation_temperature must be positive.")
        self.task = task
        self.recommendation_head = recommendation_head
        self.views = views
        self.co_feature_view = co_feature_view
        self.num_items = num_items
        self.item_feature_dims = dict(item_feature_dims or {})
        self.item_table_view = item_table_view
        self.grounding_checkpoint_path = grounding_checkpoint_path
        self.recommendation_hidden_dim = recommendation_hidden_dim
        self.recommendation_temperature = recommendation_temperature
        self.num_prompt_tokens = num_prompt_tokens
        self.freeze_backbone = freeze_backbone
        self.prompt_token_id = prompt_token_id
        self.txt_hypergraph_config = _graph_config(txt_hypergraph_config)
        self.img_hypergraph_config = _graph_config(img_hypergraph_config)
        self.ado_hypergraph_config = _graph_config(ado_hypergraph_config)
        self.vdo_hypergraph_config = _graph_config(vdo_hypergraph_config)
        self.co_hypergraph_config = _graph_config(co_hypergraph_config)

    def get_hypergraph_config(self, view: str) -> HoCRSHypergraphConfig:
        return getattr(self, f"{view}_hypergraph_config")

    def get_text_config(
        self, decoder: bool | None = None, encoder: bool | None = None
    ) -> PretrainedConfig:
        del decoder, encoder
        return self.backbone_config


__all__ = ["ALL_GRAPH_VIEWS", "HoCRSConfig", "HoCRSHypergraphConfig"]

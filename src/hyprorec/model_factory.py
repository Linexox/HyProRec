"""Model and feature-table construction for CRS training."""

from __future__ import annotations

from pathlib import Path

import torch

from .arguments import DataArguments, ModelArguments
from .backbones import BackboneAdapter
from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from .constants import MODALITIES
from .modeling_hocrs import HoCRSRecommendationModel
from .processing_hocrs import HoCRSProcessor


def load_modality_tables(
    data_args: DataArguments, model_args: ModelArguments
) -> dict[str, torch.Tensor]:
    graph_views = set(data_args.views)
    if model_args.item_table_view == "full":
        item_views = set(MODALITIES)
    elif model_args.item_table_view == "active":
        item_views = graph_views
    else:
        item_views = {model_args.item_table_view}
    needed = graph_views | item_views
    directory = Path(data_args.dataset_path) / data_args.embeddings_dir_name
    return {
        view: torch.load(
            directory / f"{view}_embeddings.pt", map_location="cpu", weights_only=True
        ).float()
        for view in MODALITIES
        if view in needed
    }


def build_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor: HoCRSProcessor,
    tables: dict[str, torch.Tensor],
    backbone_adapter: BackboneAdapter,
) -> HoCRSRecommendationModel:
    backbone_adapter.resize_token_embeddings(len(processor.tokenizer))
    backbone = backbone_adapter.model
    num_items = next(iter(tables.values())).size(0)
    graph_configs = {}
    for view in data_args.views:
        graph_configs[view] = HoCRSHypergraphConfig(
            input_dim=tables[view].size(1),
            hidden_dim=model_args.hypergraph_hidden_dim,
            output_dim=model_args.hypergraph_output_dim,
            num_layers=model_args.hypergraph_num_layers,
            dropout=model_args.hypergraph_dropout,
        )
    config = HoCRSConfig(
        backbone_config=backbone.config,
        recommendation_head=model_args.recommendation_head,
        views=data_args.views,
        num_items=num_items,
        item_feature_dims={view: table.size(1) for view, table in tables.items()},
        item_table_view=model_args.item_table_view,
        grounding_checkpoint_path=model_args.grounding_checkpoint_path,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_attention_dim=model_args.recommendation_attention_dim,
        recommendation_temperature=model_args.recommendation_temperature,
        recommendation_view_loss_weight=model_args.recommendation_view_loss_weight,
        conversation_loss_weight=model_args.conversation_loss_weight,
        recommendation_router_hidden_dim=model_args.recommendation_router_hidden_dim,
        num_prompt_tokens=model_args.num_prompt_tokens,
        freeze_backbone=model_args.freeze_backbone,
        prompt_token_id=processor.get_token_id_map()["soft_prompt_token_id"],
        **{f"{view}_hypergraph_config": value for view, value in graph_configs.items()},
    )
    model = HoCRSRecommendationModel(config, backbone_adapter=backbone_adapter)
    if model_args.grounding_checkpoint_path:
        model.load_grounding_checkpoint(model_args.grounding_checkpoint_path)
    model.initialize_feature_tables(tables)
    return model


__all__ = ["build_model", "load_modality_tables"]

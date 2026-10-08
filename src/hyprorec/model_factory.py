"""Model and feature-table construction for CRS training."""

from __future__ import annotations

from pathlib import Path

import torch

from .arguments import DataArguments, ModelArguments
from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from .constants import MODALITIES
from .modeling_hocrs import HoCRSRecommendationModel, load_backbone
from .processing_hocrs import HoCRSProcessor


def load_modality_tables(
    data_args: DataArguments, model_args: ModelArguments
) -> dict[str, torch.Tensor]:
    graph_views = {view for view in data_args.views if view in MODALITIES}
    if "co" in data_args.views:
        graph_views.add(model_args.co_feature_view)
    item_views = (
        set(MODALITIES)
        if model_args.item_table_view == "full"
        else {model_args.item_table_view}
    )
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
) -> HoCRSRecommendationModel:
    backbone = load_backbone(model_args.backbone_name_or_path)
    backbone.resize_token_embeddings(len(processor.tokenizer))
    num_items = next(iter(tables.values())).size(0)
    graph_configs = {}
    for view in data_args.views:
        feature_view = model_args.co_feature_view if view == "co" else view
        graph_configs[view] = HoCRSHypergraphConfig(
            input_dim=tables[feature_view].size(1),
            hidden_dim=model_args.hypergraph_hidden_dim,
            output_dim=model_args.hypergraph_output_dim,
            num_layers=model_args.hypergraph_num_layers,
            dropout=model_args.hypergraph_dropout,
        )
    config = HoCRSConfig(
        backbone_config=backbone.config,
        recommendation_head=model_args.recommendation_head,
        views=data_args.views,
        co_feature_view=model_args.co_feature_view,
        num_items=num_items,
        item_feature_dims={view: table.size(1) for view, table in tables.items()},
        item_table_view=model_args.item_table_view,
        grounding_checkpoint_path=model_args.grounding_checkpoint_path,
        recommendation_hidden_dim=model_args.recommendation_hidden_dim,
        recommendation_attention_dim=model_args.recommendation_attention_dim,
        recommendation_temperature=model_args.recommendation_temperature,
        recommendation_view_loss_weight=model_args.recommendation_view_loss_weight,
        use_recommendation_bpr_loss=model_args.use_recommendation_bpr_loss,
        recommendation_bpr_loss_weight=model_args.recommendation_bpr_loss_weight,
        use_recommendation_balance_loss=model_args.use_recommendation_balance_loss,
        recommendation_balance_loss_weight=model_args.recommendation_balance_loss_weight,
        conversation_loss_weight=model_args.conversation_loss_weight,
        recommendation_router_hidden_dim=model_args.recommendation_router_hidden_dim,
        num_prompt_tokens=model_args.num_prompt_tokens,
        freeze_backbone=model_args.freeze_backbone,
        prompt_token_id=processor.get_token_id_map()["soft_prompt_token_id"],
        **{f"{view}_hypergraph_config": value for view, value in graph_configs.items()},
    )
    model = HoCRSRecommendationModel(config, backbone=backbone)
    if model_args.grounding_checkpoint_path:
        model.load_grounding_checkpoint(model_args.grounding_checkpoint_path)
    model.initialize_feature_tables(tables)
    return model


__all__ = ["build_model", "load_modality_tables"]

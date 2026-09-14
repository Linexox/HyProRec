"""Ground offline graph features against trainable raw-modality encoders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import (
    AutoConfig,
    AutoModel,
    MPNetConfig,
    ViTConfig,
    VideoMAEConfig,
    Wav2Vec2Config,
)
from transformers.modeling_outputs import ModelOutput

from .configuration_grounding import HoCRSGroundingConfig
from .configuration_hocrs import HoCRSHypergraphConfig
from .losses import multi_positive_contrastive_loss
from .modeling_hocrs import HypergraphEncoder


def build_source_config(view: str):
    common = dict(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    if view == "txt":
        return MPNetConfig(**common, max_position_embeddings=64)
    if view == "img":
        return ViTConfig(**common, image_size=224, patch_size=16, num_channels=3)
    if view == "vdo":
        return VideoMAEConfig(
            **common,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=16,
            tubelet_size=2,
        )
    if view == "ado":
        common["num_hidden_layers"] = 2
        return Wav2Vec2Config(
            **common,
            conv_dim=[64] * 4,
            conv_stride=[5, 4, 4, 4],
            conv_kernel=[10, 3, 3, 3],
            num_conv_pos_embedding_groups=4,
        )
    raise ValueError(f"Unsupported source modality: {view}")


@dataclass
class HoCRSGroundingOutput(ModelOutput):
    loss: torch.Tensor | None = None
    ga_sa_loss: torch.Tensor | None = None
    ga_sn_loss: torch.Tensor | None = None
    sa_sn_loss: torch.Tensor | None = None


class HoCRSGroundingModel(PreTrainedModel):
    config_class = HoCRSGroundingConfig
    base_model_prefix = "hypergraph_encoders"
    accepts_loss_kwargs = False  # Modified: Trainer must normalize micro-batch means.

    def save_pretrained(self, save_directory, **kwargs):
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(save_directory, **kwargs)


    def __init__(self, config: HoCRSGroundingConfig) -> None:
        super().__init__(config)
        self.hypergraph_encoders = nn.ModuleDict()
        self.source_encoders = nn.ModuleDict()
        for view in config.views:
            input_dim = config.input_dims[view]
            graph_config = HoCRSHypergraphConfig(
                input_dim=input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
            self.hypergraph_encoders[view] = HypergraphEncoder(graph_config)
            if view in config.source_configs:
                source_dict = dict(config.source_configs[view])
                source_config = AutoConfig.for_model(
                    source_dict.pop("model_type"), **source_dict
                )
            else:
                source_config = build_source_config(view)
            if source_config.hidden_size != config.output_dim:
                raise ValueError("Graph output and source hidden dimensions must match.")
            config.source_configs[view] = source_config.to_dict()
            self.source_encoders[view] = AutoModel.from_config(source_config)
        self.post_init()

    def encode_source(self, view, source_data):
        output = self.source_encoders[view](**source_data).last_hidden_state
        if view in {"txt", "img"}:
            return output[:, 0]
        else:
            return output.mean(dim=1)


    def _view_loss(
        self,
        view: str,
        node_features: torch.Tensor,
        graph: Mapping[str, torch.Tensor],
        source_data: Mapping[str, torch.Tensor],  # Modified: raw modality batch.
    ) -> dict[str, torch.Tensor]:
        num_hyperedges = int(graph["edge_ptr"][-1].item())
        encoded = self.hypergraph_encoders[view](
            node_features,
            graph["hyperedge_index"],
            num_hyperedges,
        )
        source_features = self.encode_source(view, source_data).to(encoded.node_features.dtype)
        anchor_index = graph["hyperedge_anchor_index"]
        anchor_ids = graph["node_ids"].index_select(0, anchor_index)        # global node IDs
        graph_anchors = encoded.node_features.index_select(0, anchor_index)
        source_anchors = source_features.index_select(0, anchor_index)
        node_index, edge_index = graph["hyperedge_index"]
        anchor_per_incidence = anchor_index.index_select(0, edge_index)
        is_neighbor = node_index.ne(anchor_per_incidence)
        neighbor_nodes = node_index[is_neighbor]
        neighbor_edges = edge_index[is_neighbor]
        neighbor_sum = source_features.new_zeros((num_hyperedges, source_features.size(-1)))
        neighbor_sum.index_add_(0, neighbor_edges, source_features[neighbor_nodes])
        neighbor_count = torch.bincount(neighbor_edges, minlength=num_hyperedges)
        valid = neighbor_count > 0
        source_neighbors = neighbor_sum / neighbor_count.clamp_min(1).unsqueeze(-1)
        result: dict[str, torch.Tensor] = {}
        ga_sa = multi_positive_contrastive_loss(
            graph_anchors,
            source_anchors,
            anchor_ids,
            self.config.temperature
        )
        if ga_sa is not None:
            result["ga_sa"] = ga_sa
        if valid.any():
            ids = anchor_ids[valid]
            ga_sn = multi_positive_contrastive_loss(
                graph_anchors[valid],
                source_neighbors[valid],
                ids,
                self.config.temperature,
            )
            sa_sn = multi_positive_contrastive_loss(
                source_anchors[valid],
                source_neighbors[valid],
                ids,
                self.config.temperature,
            )
            if ga_sn is not None:
                result["ga_sn"] = ga_sn
            if sa_sn is not None:
                result["sa_sn"] = sa_sn
        return result

    def forward(
        self,
        node_features: Mapping[str, torch.Tensor],
        hypergraphs: Mapping[str, Mapping[str, torch.Tensor]],
        source_data: Mapping[
            str, Mapping[str, torch.Tensor]
        ],  # Modified: independent raw source branch.
        return_loss: bool = True,
        labels: torch.Tensor | None = None,
    ) -> HoCRSGroundingOutput:
        losses = {"ga_sa": [], "ga_sn": [], "sa_sn": []}
        for view in self.config.views:
            values = self._view_loss(
                view, node_features[view], hypergraphs[view], source_data[view]
            )  # Modified: preserve node ordering across branches.
            for name, value in values.items():
                losses[name].append(value)
        zero = next(iter(node_features.values())).sum() * 0.0
        terms = {
            name: torch.stack(values).mean() if values else zero
            for name, values in losses.items()
        }
        total = (
            self.config.lambda_ga_sa * terms["ga_sa"]
            + self.config.lambda_ga_sn * terms["ga_sn"]
            + self.config.lambda_sa_sn * terms["sa_sn"]
        )
        return HoCRSGroundingOutput(
            loss=total,
            ga_sa_loss=terms["ga_sa"],
            ga_sn_loss=terms["ga_sn"],
            sa_sn_loss=terms["sa_sn"],
        )


__all__ = ["HoCRSGroundingModel", "HoCRSGroundingOutput"]

"""Grounding model for node, hyperedge, and membership alignment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoTokenizer,
    MPNetConfig,
    MPNetModel,
    VideoMAEConfig,
    VideoMAEModel,
    ViTConfig,
    ViTModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

from .arguments import GroundingArguments
from .configuration_hocrs import HoCRSHypergraphConfig
from .modeling_hocrs import HypergraphEncoder


def _source_config(view: str, hidden_size: int):
    if view == "txt":
        return MPNetConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=128,
        )
    if view == "img":
        return ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            image_size=224,
            patch_size=16,
            num_channels=3,
        )
    if view == "vdo":
        return VideoMAEConfig(
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=16,
            tubelet_size=2,
        )
    if view == "ado":
        return Wav2Vec2Config(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            conv_dim=[64, 64, 64, 64],
            conv_stride=[5, 4, 4, 4],
            conv_kernel=[10, 3, 3, 3],
            num_conv_pos_embedding_groups=4,
        )
    raise ValueError(f"Unsupported grounding view: {view}")


def _contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    identifiers: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = left @ right.t() / temperature
    positives = identifiers[:, None].eq(identifiers[None, :])
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    left_loss = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    right_loss = -(
        torch.logsumexp(positive_logits.t(), dim=1)
        - torch.logsumexp(logits.t(), dim=1)
    ).mean()
    return (left_loss + right_loss) / 2


class HoCRSGroundingModel(nn.Module):
    """Train one view's graph tower against a trainable raw-content encoder."""

    def __init__(
        self, config: GroundingArguments, view: str, graph_input_dim: int | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.view = view
        graph_features_view = "txt" if view == "co" else view
        self.graph_feature_view = graph_features_view
        self.graph_feature_dim: int | None = None
        self.graph_tower: HypergraphEncoder | None = None

        source_model_name = {
            "txt": config.source_text_model,
            "img": config.source_image_model,
            "ado": config.source_audio_model,
            "vdo": config.source_video_model,
        }[graph_features_view]
        self.tokenizer = (
            AutoTokenizer.from_pretrained(source_model_name)
            if graph_features_view == "txt"
            else None
        )
        self.image_processor = (
            AutoImageProcessor.from_pretrained(source_model_name)
            if graph_features_view in {"img", "vdo"}
            else None
        )
        self.audio_processor = (
            AutoFeatureExtractor.from_pretrained(source_model_name)
            if graph_features_view == "ado"
            else None
        )
        source_config = _source_config(graph_features_view, config.source_hidden_dim)
        if graph_features_view == "txt":
            source_config.vocab_size = len(self.tokenizer)
        if graph_features_view == "txt":
            source_model = MPNetModel(source_config)
        elif graph_features_view == "img":
            source_model = ViTModel(source_config)
        elif graph_features_view == "vdo":
            source_model = VideoMAEModel(source_config)
        else:
            source_model = Wav2Vec2Model(source_config)
        self.source_encoder = source_model
        self.source_projector = nn.Linear(
            config.source_hidden_dim, config.hypergraph_output_dim
        )
        self._graph_config: HoCRSHypergraphConfig | None = None
        if graph_input_dim is not None:
            self._ensure_graph_tower(graph_input_dim)

    def _ensure_graph_tower(self, input_dim: int) -> None:
        if self.graph_tower is not None:
            if self.graph_feature_dim != input_dim:
                raise ValueError("Graph feature width changed within a run.")
            return
        self.graph_feature_dim = input_dim
        self._graph_config = HoCRSHypergraphConfig(
            input_dim=input_dim,
            hidden_dim=self.config.hypergraph_hidden_dim,
            output_dim=self.config.hypergraph_output_dim,
            num_layers=self.config.hypergraph_num_layers,
            dropout=self.config.hypergraph_dropout,
        )
        self.graph_tower = HypergraphEncoder(self._graph_config)

    def _encode_source(self, raw: Any, device: torch.device) -> torch.Tensor:
        if self.graph_feature_view == "txt":
            inputs = self.tokenizer(
                raw, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            hidden = self.source_encoder(**inputs).last_hidden_state[:, 0]
        elif self.graph_feature_view == "ado":
            values = [value.astype("float32") / 128.0 for value in raw]
            inputs = self.audio_processor(
                values, sampling_rate=16000, padding=True, return_tensors="pt"
            ).to(device)
            hidden = self.source_encoder(**inputs).last_hidden_state.mean(1)
        else:
            inputs = self.image_processor(
                [list(value) for value in raw]
                if self.graph_feature_view == "vdo"
                else list(raw),
                return_tensors="pt",
            ).to(device)
            hidden = self.source_encoder(**inputs).last_hidden_state
            hidden = (
                hidden[:, 0]
                if self.graph_feature_view == "img"
                else hidden.mean(1)
            )
        return self.source_projector(hidden)

    def _membership_loss(
        self,
        source: torch.Tensor,
        edges: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        node_ptr = graph["node_ptr"]
        edge_ptr = graph["edge_ptr"]
        node_index, edge_index = graph["hyperedge_index"]
        for start, end, edge_start, edge_end in zip(
            node_ptr[:-1], node_ptr[1:], edge_ptr[:-1], edge_ptr[1:]
        ):
            n = int(end - start)
            m = int(edge_end - edge_start)
            scores = F.normalize(source[start:end], dim=-1) @ F.normalize(
                edges[edge_start:edge_end], dim=-1
            ).t()
            scores = scores / self.config.temperature
            edge_mask = (edge_index >= edge_start) & (edge_index < edge_end)
            local_nodes = node_index[edge_mask] - start
            local_edges = edge_index[edge_mask] - edge_start
            positive = torch.zeros((n, m), dtype=torch.bool, device=scores.device)
            positive[local_nodes, local_edges] = True
            positive_scores = scores[positive]
            negative_scores = scores[~positive]
            if negative_scores.numel():
                count = min(
                    negative_scores.numel(),
                    positive_scores.numel() * self.config.member_negative_ratio,
                )
                negative_scores = negative_scores[:count]
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        torch.cat([positive_scores, negative_scores]),
                        torch.cat([
                            torch.ones_like(positive_scores),
                            torch.zeros_like(negative_scores),
                        ]),
                    )
                )
        if not losses:
            return source.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        graph = batch["hypergraphs"][self.view]
        table = batch["graph_features"][self.view]
        table = table.to(device=device)
        self._ensure_graph_tower(int(table.size(1)))
        assert self.graph_tower is not None
        encoded = self.graph_tower(
            table.index_select(0, graph["node_ids"]),
            graph["hyperedge_index"],
            int(graph["edge_ptr"][-1]),
        )
        source = self._encode_source(batch["source_data"][self.view], device)
        edge_content = source.new_zeros(encoded.hyperedge_features.shape)
        node_index, edge_index = graph["hyperedge_index"]
        edge_content.index_add_(0, edge_index, source[node_index])
        edge_degree = torch.bincount(
            edge_index, minlength=edge_content.size(0)
        ).to(source.dtype)
        edge_content = edge_content / edge_degree.clamp_min(1).unsqueeze(-1)
        node_ids = graph["node_ids"]
        edge_ids = node_ids[graph["hyperedge_anchor_index"]]
        node_loss = _contrastive_loss(
            encoded.node_features, source, node_ids, self.config.temperature
        )
        edge_loss = _contrastive_loss(
            encoded.hyperedge_features, edge_content, edge_ids, self.config.temperature
        )
        member_loss = self._membership_loss(source, encoded.hyperedge_features, graph)
        loss = (
            self.config.lambda_node * node_loss
            + self.config.lambda_edge * edge_loss
            + self.config.lambda_member * member_loss
        )
        return loss, {
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "member_loss": member_loss,
        }


__all__ = ["HoCRSGroundingModel"]

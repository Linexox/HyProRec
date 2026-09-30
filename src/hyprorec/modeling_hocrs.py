"""Decoupled recommendation and conversation models for HyProRec."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from transformers.utils import ModelOutput

from .configuration_hocrs import HoCRSConfig, canonical_feature_dims


def load_backbone(name_or_path: str) -> nn.Module:
    config = AutoConfig.from_pretrained(name_or_path)
    if config.model_type in {"qwen2_5_omni", "qwen2_5_omni_thinker"}:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            name_or_path, dtype="auto"
        )
    return AutoModelForCausalLM.from_pretrained(name_or_path)


def _backbone_from_config(config) -> nn.Module:
    if config.model_type == "qwen2_5_omni_thinker":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        return Qwen2_5OmniThinkerForConditionalGeneration(config)
    return AutoModelForCausalLM.from_config(config)


def _hidden_size(config) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        if getattr(config, name, None) is not None:
            return int(getattr(config, name))
    return int(config.text_config.hidden_size)


@dataclass
class HypergraphEncoderOutput:
    node_features: torch.Tensor
    hyperedge_features: torch.Tensor


def hypergraph_propagate(
    node_features: torch.Tensor, hyperedge_index: torch.Tensor, num_hyperedges: int
) -> HypergraphEncoderOutput:
    node_index, edge_index = hyperedge_index
    dtype = node_features.dtype
    node_degree = torch.bincount(node_index, minlength=node_features.size(0)).to(
        dtype=dtype
    )
    edge_degree = torch.bincount(edge_index, minlength=num_hyperedges).to(dtype=dtype)
    node_scale = node_degree.clamp_min(1).pow(-0.5)
    edge_scale = edge_degree.clamp_min(1).reciprocal()
    normalized_nodes = node_scale.unsqueeze(-1) * node_features
    edge_features = node_features.new_zeros((num_hyperedges, node_features.size(-1)))
    edge_features.index_add_(0, edge_index, normalized_nodes[node_index])
    edge_features = edge_scale.unsqueeze(-1) * edge_features
    propagated = node_features.new_zeros(node_features.shape)
    propagated.index_add_(0, node_index, edge_features[edge_index])
    return HypergraphEncoderOutput(node_scale.unsqueeze(-1) * propagated, edge_features)


class HypergraphEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        dimensions = [
            config.input_dim,
            *([config.hidden_dim] * (config.num_layers - 1)),
            config.output_dim,
        ]
        self.layers = nn.ModuleList(
            nn.Linear(a, b, bias=config.use_bias)
            for a, b in zip(dimensions, dimensions[1:])
        )
        self.norms = nn.ModuleList(nn.LayerNorm(size) for size in dimensions[1:])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, node_features, hyperedge_index, num_hyperedges):
        output = None
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            transformed = norm(layer(node_features))
            if index + 1 < len(self.layers):
                transformed = self.dropout(F.gelu(transformed))
            output = hypergraph_propagate(transformed, hyperedge_index, num_hyperedges)
            node_features = transformed + output.node_features
        return HypergraphEncoderOutput(node_features, output.hyperedge_features)


class HypergraphProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.node_projector = nn.Linear(input_dim, output_dim)
        self.hyperedge_projector = nn.Linear(input_dim, output_dim)

    def forward(self, output):
        return HypergraphEncoderOutput(
            self.node_projector(output.node_features),
            self.hyperedge_projector(output.hyperedge_features),
        )


class GraphTokenMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        views: tuple[str, ...],
        expert_hidden_size: int = 512,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.view_to_index = {view: i for i, view in enumerate(views)}
        self.view_embeddings = nn.Parameter(torch.randn(len(views), hidden_size) * 0.02)
        self.router = nn.Linear(hidden_size * 2, num_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, expert_hidden_size),
                nn.GELU(),
                nn.Linear(expert_hidden_size, hidden_size),
            )
            for _ in range(num_experts)
        )
        self.residual_scales = nn.Parameter(torch.ones(len(views)))

    def reset_output_layers(self) -> None:
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, features: torch.Tensor, view: str) -> torch.Tensor:
        view_embedding = self.view_embeddings[self.view_to_index[view]].expand(
            features.size(0), -1
        )
        weights = F.softmax(
            self.router(
                torch.cat((features, view_embedding.to(features.dtype)), dim=-1).float()
            ),
            dim=-1,
        )
        outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        delta = (weights.to(outputs.dtype).unsqueeze(-1) * outputs).sum(dim=1)
        return (
            features
            + self.residual_scales[self.view_to_index[view]].to(features.dtype) * delta
        )


class ItemTableHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dims: dict[str, int],
        item_view: str,
        output_dim: int,
        temperature: float,
        router_hidden_dim: int,
        view_loss_weight: float,
        use_balance_loss: bool,
        balance_loss_weight: float,
    ) -> None:
        super().__init__()
        self.item_view = item_view
        self.temperature = temperature
        self.view_loss_weight = view_loss_weight
        self.use_balance_loss = use_balance_loss
        self.balance_loss_weight = balance_loss_weight
        feature_dims = canonical_feature_dims(feature_dims)
        names = tuple(feature_dims) if item_view == "full" else (item_view,)
        self.names = names
        self.user_projects = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                )
                for name in names
            }
        )
        self.item_projects = nn.ModuleDict(
            {
                name: nn.Linear(feature_dims[name], output_dim, bias=False)
                for name in names
            }
        )
        self.router = nn.Sequential(
            nn.Linear(output_dim * len(names), router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, len(names)),
        )

    def item_embeddings(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: F.normalize(
                self.item_projects[name](features[name].float()), dim=-1
            )
            for name in self.names
        }

    def forward(
        self,
        pooled: torch.Tensor,
        features: Mapping[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> "RecommendationHeadOutput":
        users = {
            name: F.normalize(self.user_projects[name](pooled.float()), dim=-1)
            for name in self.names
        }
        items = self.item_embeddings(features)
        view_logits = tuple(
            users[name] @ items[name].t() / self.temperature for name in self.names
        )
        route_input = torch.cat([users[name] for name in self.names], dim=-1)
        route_weights = F.softmax(self.router(route_input.float()), dim=-1)
        stacked_logits = torch.stack(view_logits, dim=1)
        fusion_logits = (
            stacked_logits * route_weights.to(stacked_logits.dtype).unsqueeze(-1)
        ).sum(dim=1)

        fusion_loss = view_loss = balance_loss = None
        loss = None
        if labels is not None:
            fusion_loss = F.cross_entropy(fusion_logits, labels)
            view_loss = (
                torch.stack(
                    [F.cross_entropy(logits, labels) for logits in view_logits]
                ).mean()
                if len(self.names) > 1
                else fusion_loss.new_zeros(())
            )
            if self.use_balance_loss:
                mean_route = route_weights.float().mean(dim=0).clamp_min(1e-8)
                balance_loss = (
                    mean_route * (mean_route.log() + log(len(self.names)))
                ).sum()
            else:
                balance_loss = fusion_logits.new_zeros(())
            loss = fusion_loss + self.view_loss_weight * view_loss
            if self.use_balance_loss:
                loss = loss + self.balance_loss_weight * balance_loss

        return RecommendationHeadOutput(
            logits=fusion_logits,
            loss=loss,
            fusion_loss=fusion_loss,
            view_loss=view_loss,
            balance_loss=balance_loss,
            route_weights=route_weights,
            view_logits=view_logits,
        )


@dataclass
class RecommendationHeadOutput(ModelOutput):
    """Recommendation logits and diagnostics exposed to Trainer and callbacks."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    fusion_loss: torch.Tensor | None = None
    view_loss: torch.Tensor | None = None
    balance_loss: torch.Tensor | None = None
    route_weights: torch.Tensor | None = None
    view_logits: tuple[torch.Tensor, ...] | None = None


@dataclass
class HoCRSRecommendationOutput(SequenceClassifierOutput):
    """Trainer-compatible output with per-view recommendation diagnostics."""

    fusion_loss: torch.Tensor | None = None
    view_loss: torch.Tensor | None = None
    balance_loss: torch.Tensor | None = None
    route_weights: torch.Tensor | None = None
    view_logits: tuple[torch.Tensor, ...] | None = None


class RecommendationHead(nn.Module):
    def __init__(self, input_dim: int, config: HoCRSConfig) -> None:
        super().__init__()
        self.head = ItemTableHead(
            input_dim,
            config.item_feature_dims,
            config.item_table_view,
            config.recommendation_hidden_dim,
            config.recommendation_temperature,
            config.recommendation_router_hidden_dim,
            config.recommendation_view_loss_weight,
            config.use_recommendation_balance_loss,
            config.recommendation_balance_loss_weight,
        )

    def forward(self, pooled, features, labels=None):
        return self.head(pooled, features, labels)


class HoCRSBaseModel(PreTrainedModel, GenerationMixin):
    config_class = HoCRSConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    def __init__(self, config: HoCRSConfig, backbone: nn.Module | None = None) -> None:
        super().__init__(config)
        self.backbone = backbone or _backbone_from_config(config.backbone_config)
        hidden_size = _hidden_size(config.backbone_config)
        self.soft_prompt_embeddings = nn.Embedding(
            config.num_prompt_tokens, hidden_size
        )
        self.hypergraph_encoders = nn.ModuleDict()
        self.hypergraph_projectors = nn.ModuleDict()
        needed = {view for view in config.views if view != "co"}
        if "co" in config.views:
            needed.add(config.co_feature_view)
        # Keep graph-side feature buffers for all loaded item modalities for
        # checkpoint/API compatibility.  Recommendation reads the separate
        # ``item_*_feature_table`` buffers below, so the two paths remain
        # independent despite identical initialization values.
        if config.task == "recommendation":
            needed.update(config.item_feature_dims)
        for view in needed:
            self.register_buffer(
                f"{view}_feature_table",
                torch.zeros(config.num_items, config.item_feature_dims[view]),
                persistent=True,
            )
        if config.task == "recommendation":
            for view in config.item_feature_dims:
                self.register_buffer(
                    f"item_{view}_feature_table",
                    torch.zeros(config.num_items, config.item_feature_dims[view]),
                    persistent=True,
                )
        for view in config.views:
            graph_config = config.get_hypergraph_config(view)
            self.hypergraph_encoders[view] = HypergraphEncoder(graph_config)
            self.hypergraph_projectors[view] = HypergraphProjector(
                graph_config.output_dim, hidden_size
            )
        self.graph_moe = GraphTokenMoE(hidden_size, config.views)
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.post_init()
        self.graph_moe.reset_output_layers()

    def initialize_feature_tables(self, tables: Mapping[str, torch.Tensor]) -> None:
        for name, table in tables.items():
            value = table.float()
            graph_table = getattr(self, f"{name}_feature_table", None)
            if graph_table is not None:
                graph_table.copy_(value)
            item_table = getattr(self, f"item_{name}_feature_table", None)
            if item_table is not None:
                item_table.copy_(value)

    def _feature_tables(self) -> dict[str, torch.Tensor]:
        names = (
            self.config.item_feature_dims
            if self.config.item_table_view == "full"
            else (self.config.item_table_view,)
        )
        return {
            name: getattr(self, f"item_{name}_feature_table")
            for name in names
        }

    def load_grounding_checkpoint(self, path: str) -> None:
        directory = Path(path)
        if (directory / "model.safetensors").exists():
            from safetensors.torch import load_file

            state = load_file(str(directory / "model.safetensors"), device="cpu")
        else:
            state = torch.load(directory / "pytorch_model.bin", map_location="cpu")
        for view, encoder in self.hypergraph_encoders.items():
            grounding_view = (
                f"co_{self.config.co_feature_view}" if view == "co" else view
            )
            prefix = f"hypergraph_encoders.{grounding_view}."
            encoder.load_state_dict(
                {
                    key.removeprefix(prefix): value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
            )
            encoder.requires_grad_(False)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.backbone.set_output_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def _inject(self, input_ids, hypergraphs):
        embeds = self.get_input_embeddings()(input_ids).clone()
        prompt_positions = input_ids == self.config.prompt_token_id
        prompt_ids = torch.arange(
            self.config.num_prompt_tokens, device=input_ids.device
        ).expand(input_ids.size(0), -1)
        embeds[prompt_positions] = (
            self.soft_prompt_embeddings(prompt_ids)
            .reshape(-1, embeds.size(-1))
            .to(embeds.dtype)
        )
        for view, graph in hypergraphs.items():
            feature_view = self.config.co_feature_view if view == "co" else view
            node_features = getattr(self, f"{feature_view}_feature_table").index_select(0, graph["node_ids"])
            encoded = self.hypergraph_encoders[view](node_features, graph["hyperedge_index"], int(graph["edge_ptr"][-1]))
            projected_graph = self.hypergraph_projectors[view](encoded)
            projected = self.graph_moe(
                view=view,
                features=torch.cat(
                    (projected_graph.node_features, projected_graph.hyperedge_features)
                ),
            )
            node_count = graph["node_positions"].size(0)
            embeds[graph["node_positions"][:, 0], graph["node_positions"][:, 1]] = projected[:node_count].to(embeds.dtype)
            embeds[graph["hyperedge_positions"][:, 0], graph["hyperedge_positions"][:, 1]] = projected[node_count:].to(embeds.dtype)
        return embeds

    def encode(self, input_ids, attention_mask=None, hypergraphs=None, **kwargs):
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        cache = kwargs.get("past_key_values")
        cached_step = cache is not None and cache.get_seq_length() > 0
        embeds = (
            self.get_input_embeddings()(input_ids)
            if cached_step
            else self._inject(input_ids, hypergraphs or {})
        )
        if self.config.backbone_config.model_type == "qwen2_5_omni_thinker":
            kwargs["input_ids"] = input_ids
        return self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )


class HoCRSRecommendationModel(HoCRSBaseModel):
    def __init__(self, config, backbone=None):
        super().__init__(config, backbone)
        self.recommendation_head = RecommendationHead(
            _hidden_size(config.backbone_config), config
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pooling_mask=None,
        rec_labels=None,
        hypergraphs=None,
        **kwargs,
    ):
        output = self.encode(input_ids, attention_mask, hypergraphs, **kwargs)
        hidden = output.hidden_states[-1]
        # DialoGPT runs in float16.  Accumulating a long sequence in float16
        # can overflow even when each hidden state is finite, so pool in fp32.
        mask = pooling_mask.to(device=hidden.device, dtype=torch.float32)
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        recommendation = self.recommendation_head(
            pooled, self._feature_tables(), labels=rec_labels
        )
        return HoCRSRecommendationOutput(
            loss=recommendation.loss,
            logits=recommendation.logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            fusion_loss=recommendation.fusion_loss,
            view_loss=recommendation.view_loss,
            balance_loss=recommendation.balance_loss,
            route_weights=recommendation.route_weights,
            view_logits=recommendation.view_logits,
        )


class HoCRSConversationModel(HoCRSBaseModel):
    def forward(
        self, input_ids, attention_mask=None, labels=None, hypergraphs=None, **kwargs
    ):
        output = self.encode(input_ids, attention_mask, hypergraphs, labels=labels, **kwargs)
        return CausalLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


__all__ = [
    "HoCRSConfig",
    "HoCRSConversationModel",
    "HoCRSRecommendationModel",
    "HypergraphEncoder",
    "HypergraphEncoderOutput",
    "HypergraphProjector",
    "GraphTokenMoE",
    "ItemTableHead",
    "RecommendationHead",
    "RecommendationHeadOutput",
    "HoCRSRecommendationOutput",
    "hypergraph_propagate",
    "load_backbone",
]

"""Hypergraph-prompted recommendation and teacher-forced dialogue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput

from .configuration_hocrs import HoCRSConfig, canonical_feature_dims
from .backbones import BackboneAdapter, adapter_from_config, adapter_from_model


@dataclass
class HypergraphEncoderOutput:
    node_features: torch.Tensor
    hyperedge_features: torch.Tensor


def hypergraph_propagate(
    node_features: torch.Tensor, hyperedge_index: torch.Tensor, num_hyperedges: int
) -> HypergraphEncoderOutput:
    node_index, edge_index = hyperedge_index
    dtype = node_features.dtype
    node_degree = torch.bincount(node_index, minlength=node_features.size(0)).to(dtype=dtype)
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
        self.num_experts = num_experts
        self.view_embeddings = nn.Parameter(torch.randn(len(views), hidden_size) * 0.02)
        self.router = nn.Linear(hidden_size * 2, num_experts)
        self.expert_in_weight = nn.Parameter(
            torch.empty(num_experts, expert_hidden_size, hidden_size)
        )
        self.expert_in_bias = nn.Parameter(torch.zeros(num_experts, expert_hidden_size))
        self.expert_out_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, expert_hidden_size)
        )
        self.expert_out_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))
        nn.init.xavier_uniform_(self.expert_in_weight)
        nn.init.xavier_uniform_(self.expert_out_weight)
        self.residual_scales = nn.Parameter(torch.ones(len(views)))

    def reset_output_layers(self) -> None:
        nn.init.zeros_(self.expert_out_weight)
        nn.init.zeros_(self.expert_out_bias)

    def forward(
        self,
        features: torch.Tensor,
        view: str,
        return_diagnostics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        view_index = self.view_to_index[view]
        view_embedding = self.view_embeddings[view_index]
        view_embedding = view_embedding.expand(features.size(0), -1)
        router_input = torch.cat((features, view_embedding.to(features.dtype)), dim=-1)
        router_input = router_input.float()
        expert_hidden = torch.bmm(
            features.unsqueeze(0).expand(self.num_experts, -1, -1),
            self.expert_in_weight.transpose(1, 2),
        ).permute(1, 0, 2)
        expert_hidden = expert_hidden + self.expert_in_bias.unsqueeze(0)
        expert_hidden = F.gelu(expert_hidden)
        outputs = torch.bmm(
            expert_hidden.permute(1, 0, 2),
            self.expert_out_weight.transpose(1, 2),
        ).permute(1, 0, 2)
        outputs = outputs + self.expert_out_bias.unsqueeze(0)
        weights = F.softmax(self.router(router_input), dim=-1)
        delta = (weights.to(outputs.dtype).unsqueeze(-1) * outputs).sum(dim=1)
        scale = self.residual_scales[view_index].to(features.dtype)
        transformed = features + scale * delta

        if return_diagnostics:
            return transformed, weights
        else:
            return transformed


class ViewAttentionPooling(nn.Module):
    """Attend to backbone tokens using a learned token for this view."""

    def __init__(self, hidden_size: int, attention_dim: int) -> None:
        super().__init__()
        self.attention_view_token = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.query = nn.Linear(hidden_size, attention_dim, bias=False)
        self.key = nn.Linear(hidden_size, attention_dim, bias=False)
        self.value = nn.Linear(hidden_size, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        pooling_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden.float()
        mask = pooling_mask.to(device=hidden.device, dtype=torch.bool)
        query = self.query(self.attention_view_token.float())
        keys = self.key(hidden)
        values = self.value(hidden)
        scores = (query * keys).sum(dim=-1) / (keys.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        attended = (weights.unsqueeze(-1) * values).sum(dim=1)
        return self.output(attended), weights


class ItemTableHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dims: dict[str, int],
        item_view: str,
        output_dim: int,
        attention_dim: int,
        temperature: float,
        router_hidden_dim: int,
        view_loss_weight: float,
    ) -> None:
        super().__init__()
        self.item_view = item_view
        self.temperature = temperature
        self.view_loss_weight = view_loss_weight
        feature_dims = canonical_feature_dims(feature_dims)
        names = tuple(feature_dims) if item_view in {"full", "active"} else (item_view,)
        self.names = names
        self.attention_poolers = nn.ModuleDict({
            name: ViewAttentionPooling(input_dim, attention_dim)
            for name in names
        })
        self.user_projects = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
            for name in names
        })
        self.item_projects = nn.ModuleDict({
            name: nn.Linear(feature_dims[name], output_dim, bias=False)
            for name in names
        })
        self.router = nn.Sequential(
            nn.Linear(output_dim * len(names), router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, len(names)),
        )

    def item_embeddings(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: F.normalize(
                self.item_projects[name](features[name].float()),
                dim=-1
            )
            for name in self.names
        }

    def forward(
        self,
        hidden: torch.Tensor,
        pooling_mask: torch.Tensor,
        features: Mapping[str, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> "RecommendationHeadOutput":
        pooled = {}
        attention_weights = {}
        users = {}
        for name in self.names:
            pooled[name], attention_weights[name] = self.attention_poolers[name](
                hidden,
                pooling_mask,
            )
            users[name] = F.normalize(self.user_projects[name](pooled[name].float()), dim=-1)
        items = self.item_embeddings(features)
        view_logits = tuple(
            users[name] @ items[name].t() / self.temperature
            for name in self.names
        )
        route_input = torch.cat([users[name] for name in self.names], dim=-1)
        route_weights = F.softmax(self.router(route_input.float()), dim=-1)
        stacked_logits = torch.stack(view_logits, dim=1)
        fusion_logits = (
            stacked_logits * route_weights.to(stacked_logits.dtype).unsqueeze(-1)
        ).sum(dim=1)

        fusion_loss = view_loss = None
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
            loss = fusion_loss + self.view_loss_weight * view_loss

        return RecommendationHeadOutput(
            logits=fusion_logits,
            loss=loss,
            fusion_loss=fusion_loss,
            view_loss=view_loss,
            route_weights=route_weights,
            view_logits=view_logits,
            attention_weights=attention_weights,
        )


@dataclass
class RecommendationHeadOutput(ModelOutput):
    """Recommendation logits and diagnostics exposed to Trainer and callbacks."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    fusion_loss: torch.Tensor | None = None
    view_loss: torch.Tensor | None = None
    route_weights: torch.Tensor | None = None
    view_logits: tuple[torch.Tensor, ...] | None = None
    attention_weights: dict[str, torch.Tensor] | None = None


@dataclass
class HoCRSRecommendationOutput(SequenceClassifierOutput):
    """Recommendation output, optionally carrying teacher-forced LM outputs."""

    recommendation_logits: torch.Tensor | None = None

    fusion_loss: torch.Tensor | None = None
    view_loss: torch.Tensor | None = None
    conversation_loss: torch.Tensor | None = None
    conversation_logits: torch.Tensor | None = None
    conversation_predictions: torch.Tensor | None = None
    route_weights: torch.Tensor | None = None
    view_logits: tuple[torch.Tensor, ...] | None = None
    attention_weights: dict[str, torch.Tensor] | None = None
    graph_moe_diagnostics: dict[str, dict[str, torch.Tensor]] | None = None


class HoCRSBaseModel(PreTrainedModel, GenerationMixin):
    config_class = HoCRSConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    @classmethod
    def _fix_state_dict_key_on_load(cls, key: str) -> tuple[str, bool]:
        key, changed = super()._fix_state_dict_key_on_load(key)
        key, count = re.subn(
            r"^recommendation_head\.head\.(.*)$", r"recommendation_head.\1", key
        )
        return key, changed or count > 0

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load pre-cleanup checkpoints whose head had one redundant wrapper."""

        remapped = state_dict.copy()
        prefix = "recommendation_head.head."
        for key in list(remapped):
            if key.startswith(prefix):
                remapped[key.removeprefix(prefix)] = remapped.pop(key)
        return super().load_state_dict(remapped, strict=strict, assign=assign)

    def __init__(
        self,
        config: HoCRSConfig,
        backbone: nn.Module | None = None,
        backbone_adapter: BackboneAdapter | None = None,
    ) -> None:
        super().__init__(config)
        if backbone_adapter is None:
            backbone_adapter = (
                adapter_from_model(backbone)
                if backbone is not None
                else adapter_from_config(config.backbone_config)
            )
        self.backbone_adapter = backbone_adapter
        self.backbone = backbone_adapter.model
        hidden_size = backbone_adapter.hidden_size
        self.soft_prompt_embeddings = nn.Embedding(config.num_prompt_tokens, hidden_size)
        self.hypergraph_encoders = nn.ModuleDict()
        self.hypergraph_projectors = nn.ModuleDict()
        needed = set(config.views)
        # Keep graph-side feature buffers for all loaded item modalities for
        # checkpoint/API compatibility.  Recommendation reads the separate
        # ``item_*_feature_table`` buffers below, so the two paths remain
        # independent despite identical initialization values.
        needed.update(config.item_feature_dims)
        for view in sorted(needed):
            self.register_buffer(
                f"{view}_feature_table",
                torch.zeros(config.num_items, config.item_feature_dims[view]),
                persistent=True,
            )
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
        return {
            name: getattr(self, f"item_{name}_feature_table")
            for name in self.config.item_views
        }

    def load_grounding_checkpoint(self, path: str) -> None:
        directory = Path(path)
        if (directory / "model.safetensors").exists():
            from safetensors.torch import load_file

            state = load_file(str(directory / "model.safetensors"), device="cpu")
        else:
            state = torch.load(directory / "pytorch_model.bin", map_location="cpu")
        for view, encoder in self.hypergraph_encoders.items():
            grounding_view = view
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
        return self.backbone_adapter.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone_adapter.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.backbone_adapter.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.backbone_adapter.set_output_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone_adapter.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone_adapter.gradient_checkpointing_disable()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            if mode and self.backbone.is_gradient_checkpointing:
                # Checkpointed decoder layers need training mode even when
                # their parameters are frozen. Keep frozen-backbone dropout off.
                for module in self.backbone.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()
            else:
                self.backbone.eval()
        return self

    def _inject(self, input_ids, hypergraphs):
        embeds = self.get_input_embeddings()(input_ids).clone()
        prompt_positions = input_ids == self.config.prompt_token_id
        prompt_ids = (
            torch.arange(self.config.num_prompt_tokens)
            .expand(input_ids.size(0), -1)
            .to(input_ids.device)
        )
        embeds[prompt_positions] = (
            self.soft_prompt_embeddings(prompt_ids)
            .reshape(-1, embeds.size(-1))
            .to(embeds.dtype)
        )
        for view, graph in hypergraphs.items():
            node_features = (
                getattr(self, f"{view}_feature_table")
                .index_select(0, graph["node_ids"])
            )
            encoded = self.hypergraph_encoders[view](
                node_features,
                graph["hyperedge_index"],
                int(graph["edge_ptr"][-1]),
            )
            projected_graph = self.hypergraph_projectors[view](encoded)
            projected = self.graph_moe(
                view=view,
                features=torch.cat((
                    projected_graph.node_features,
                    projected_graph.hyperedge_features
                )),
            )
            node_count = graph["node_positions"].size(0)
            node_positions = graph["node_positions"]
            hyperedge_positions = graph["hyperedge_positions"]
            projected = projected.to(embeds.dtype)
            embeds[node_positions[:, 0], node_positions[:, 1]]              = projected[:node_count]
            embeds[hyperedge_positions[:, 0], hyperedge_positions[:, 1]]    = projected[node_count:]
        return embeds

    def encode(self, input_ids, attention_mask=None, hypergraphs=None, **kwargs):
        kwargs.pop("return_dict", None)
        kwargs.pop("inputs_embeds", None)
        requested_hidden_states = kwargs.pop("output_hidden_states", False)
        labels      = kwargs.pop("labels", None)
        use_cache   = kwargs.pop("use_cache", not self.training)
        cache       = kwargs.pop("past_key_values", None)

        cached_step = cache is not None and cache.get_seq_length() > 0
        if cached_step:
            self._last_graph_moe_diagnostics = {}
        embeds = (
            self.get_input_embeddings()(input_ids)
            if cached_step
            else self._inject(input_ids, hypergraphs or {})
        )

        decoder_output = self.backbone_adapter.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=embeds,
            past_key_values=cache,
            output_hidden_states=requested_hidden_states,
            use_cache=use_cache,
            **kwargs,
        )
        hidden = self.backbone_adapter.get_last_hidden_state(decoder_output)
        logits = None
        loss = None
        predictions = None
        if labels is not None:
            targets = labels[..., 1:]
            supervised = targets.ne(-100)
            selected_hidden = hidden[..., :-1, :][supervised]
            selected_logits = self.backbone_adapter.compute_lm_logits(selected_hidden)
            loss = F.cross_entropy(
                selected_logits.float(),
                targets[supervised],
            )
            if not self.training:
                predictions = targets.new_full(targets.shape, -100)
                predictions[supervised] = selected_logits.argmax(dim=-1)
        else:
            logits = self.backbone_adapter.compute_lm_logits(hidden)
        hidden_states = decoder_output.hidden_states
        if hidden_states is None:
            hidden_states = (hidden,)
        output = ModelOutput(
            loss=loss,
            logits=logits,
            conversation_predictions=predictions,
            **self.backbone_adapter.prepare_generation_cache(decoder_output),
            hidden_states=hidden_states,
            attentions=getattr(decoder_output.raw, "attentions", None),
        )
        return output


class HoCRSRecommendationModel(HoCRSBaseModel):
    def __init__(self, config, backbone=None, backbone_adapter=None):
        super().__init__(config, backbone, backbone_adapter)
        self.recommendation_head = ItemTableHead(
            self.backbone_adapter.hidden_size,
            {view: config.item_feature_dims[view] for view in config.item_views},
            config.item_table_view,
            config.recommendation_hidden_dim,
            config.recommendation_attention_dim,
            config.recommendation_temperature,
            config.recommendation_router_hidden_dim,
            config.recommendation_view_loss_weight,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pooling_mask=None,
        rec_labels=None,
        labels=None,
        hypergraphs=None,
        **kwargs,
    ):
        output = self.encode(
            input_ids,
            attention_mask,
            hypergraphs,
            labels=labels,
            **kwargs,
        )
        hidden = output.hidden_states[-1]
        if pooling_mask is None:
            pooling_mask = attention_mask.bool()
        recommendation = self.recommendation_head(
            hidden, pooling_mask, self._feature_tables(), labels=rec_labels
        )
        zero = hidden.new_zeros(())
        recommendation_loss = recommendation.loss if recommendation.loss is not None else zero
        conversation_loss = output.loss if output.loss is not None else zero
        loss = recommendation_loss + self.config.conversation_loss_weight * conversation_loss
        return HoCRSRecommendationOutput(
            loss=loss,
            logits=output.logits,
            recommendation_logits=recommendation.logits,
            fusion_loss=recommendation.fusion_loss,
            view_loss=recommendation.view_loss,
            conversation_loss=output.loss,
            # Evaluation needs teacher-forced predictions; training only needs
            # the LM loss. Only response positions enter the vocabulary head.
            conversation_logits=output.logits,
            conversation_predictions=output.conversation_predictions,
            route_weights=recommendation.route_weights,
            view_logits=recommendation.view_logits,
            attention_weights=recommendation.attention_weights,
            graph_moe_diagnostics=getattr(self, "_last_graph_moe_diagnostics", {}),
        )


__all__ = [
    "HoCRSConfig",
    "HoCRSRecommendationModel",
    "HypergraphEncoder",
    "HypergraphEncoderOutput",
    "HypergraphProjector",
    "GraphTokenMoE",
    "ItemTableHead",
    "ViewAttentionPooling",
    "RecommendationHeadOutput",
    "HoCRSRecommendationOutput",
    "hypergraph_propagate",
]

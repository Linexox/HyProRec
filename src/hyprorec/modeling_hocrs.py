"""Decoupled recommendation and conversation models for HyProRec."""

from __future__ import annotations

from dataclasses import dataclass
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

from .configuration_hocrs import HoCRSConfig


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
    ) -> None:
        super().__init__()
        self.item_view = item_view
        self.temperature = temperature
        names = tuple(feature_dims) if item_view == "full" else (item_view,)
        self.names = names
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.projects = nn.ModuleDict(
            {
                name: nn.Linear(feature_dims[name], output_dim, bias=False)
                for name in names
            }
        )
        self.fuse = (
            nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
            if len(names) > 1
            else None
        )

    def item_embeddings(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        projected = torch.stack(
            [self.projects[name](features[name]) for name in self.names], dim=1
        )
        if self.fuse is not None:
            projected = self.fuse(projected, projected, projected, need_weights=False)[
                0
            ]
        return F.normalize(projected.mean(dim=1), dim=-1)

    def forward(
        self, pooled: torch.Tensor, features: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return (
            F.normalize(self.query(pooled), dim=-1)
            @ self.item_embeddings(features).t()
            / self.temperature
        )


class GlobalHyperedgeMixtureHead(nn.Module):
    """Softly mix per-view global hyperedge distributions."""

    def __init__(self, input_dim, output_dim, views, temperature):
        super().__init__()
        self.views = tuple(views)
        self.temperature = temperature
        self.user_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.table_projectors = nn.ModuleDict(
            {view: nn.Linear(input_dim, output_dim, bias=False) for view in self.views}
        )
        self.router = nn.Linear(input_dim, len(self.views))

    def forward(self, pooled, tables):
        user = F.normalize(self.user_projector(pooled), dim=-1)
        probabilities = []
        for view in self.views:
            table = F.normalize(self.table_projectors[view](tables[view]), dim=-1)
            probabilities.append((user @ table.t() / self.temperature).softmax(dim=-1))
        stacked = torch.stack(probabilities, dim=1)
        route = self.router(pooled.float()).softmax(dim=-1).to(stacked.dtype)
        return (
            torch.log((stacked * route.unsqueeze(-1)).sum(dim=1).clamp_min(1e-8)),
            route,
        )


class RecommendationHead(nn.Module):
    def __init__(self, input_dim: int, config: HoCRSConfig) -> None:
        super().__init__()
        if config.recommendation_head == "item_table":
            self.head = ItemTableHead(
                input_dim,
                config.item_feature_dims,
                config.item_table_view,
                config.recommendation_hidden_dim,
                config.recommendation_temperature,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(input_dim, config.recommendation_hidden_dim),
                nn.GELU(),
                nn.Linear(config.recommendation_hidden_dim, config.num_items),
            )

    def forward(self, pooled, features):
        if isinstance(self.head, ItemTableHead):
            return self.head(pooled, features)
        return self.head(pooled)


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
        needed = set(config.item_feature_dims)
        needed.update(view for view in config.views if view != "co")
        if "co" in config.views:
            needed.add(config.co_feature_view)
        for view in needed:
            self.register_buffer(
                f"{view}_feature_table",
                torch.zeros(config.num_items, config.item_feature_dims[view]),
                persistent=True,
            )
        for view in config.views:
            graph_config = config.get_hypergraph_config(view)
            self.hypergraph_encoders[view] = HypergraphEncoder(graph_config)
            self.hypergraph_projectors[view] = HypergraphProjector(
                graph_config.output_dim, hidden_size
            )
        self.graph_moe = (
            None
            if config.global_hypergraph
            else GraphTokenMoE(hidden_size, config.views)
        )
        self._global_hypergraph_outputs = {}
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.post_init()
        if self.graph_moe is not None:
            self.graph_moe.reset_output_layers()

    def initialize_feature_tables(self, tables: Mapping[str, torch.Tensor]) -> None:
        for name, table in tables.items():
            getattr(self, f"{name}_feature_table").copy_(table.float())

    def _feature_tables(self) -> dict[str, torch.Tensor]:
        if self.config.recommendation_head == "mlp":
            return {}
        names = (
            self.config.item_feature_dims
            if self.config.item_table_view == "full"
            else (self.config.item_table_view,)
        )
        return {name: getattr(self, f"{name}_feature_table") for name in names}

    def load_grounding_checkpoint(self, path: str) -> None:
        directory = Path(path)
        if (directory / "model.safetensors").exists():
            from safetensors.torch import load_file

            state = load_file(str(directory / "model.safetensors"), device="cpu")
        else:
            state = torch.load(directory / "pytorch_model.bin", map_location="cpu")
        for view, encoder in self.hypergraph_encoders.items():
            candidates = [view]
            if view == "co":
                candidates.extend(
                    [f"co_{self.config.co_feature_view}", self.config.co_feature_view]
                )
            for grounding_view in candidates:
                prefix = f"hypergraph_encoders.{grounding_view}."
                values = {
                    key.removeprefix(prefix): value
                    for key, value in state.items()
                    if key.startswith(prefix)
                }
                if values:
                    encoder.load_state_dict(values, strict=True)
                    break
            else:
                raise KeyError(f"No Grounding tower found for CRS view '{view}'.")
            encoder.requires_grad_(False)
            encoder.eval()

    @torch.no_grad()
    def cache_global_hypergraph_towers(self, hypergraphs, feature_tables=None) -> None:
        """Encode each full-catalog graph once; only projectors train afterward."""
        device = next(self.parameters()).device
        feature_tables = feature_tables or {
            name: getattr(self, f"{name}_feature_table")
            for name in set(self.config.item_feature_dims)
            | {self.config.co_feature_view}
            if hasattr(self, f"{name}_feature_table")
        }
        self._global_hypergraph_outputs = {}
        for view, graph in hypergraphs.items():
            feature_view = (
                view if view in feature_tables else self.config.co_feature_view
            )
            node_ids = graph.node_ids.to(device=device)
            node_features = (
                feature_tables[feature_view].to(device=device).index_select(0, node_ids)
            )
            encoder = self.hypergraph_encoders[view]
            encoder.requires_grad_(False)
            encoder.eval()
            output = encoder(
                node_features,
                graph.hyperedge_index.to(device=device),
                graph.num_hyperedges,
            )
            self._global_hypergraph_outputs[view] = HypergraphEncoderOutput(
                output.node_features.detach(), output.hyperedge_features.detach()
            )

    def project_global_hyperedges(self):
        if not self._global_hypergraph_outputs:
            raise RuntimeError("Global hypergraph towers have not been cached.")
        return {
            view: self.hypergraph_projectors[view](output).hyperedge_features
            for view, output in self._global_hypergraph_outputs.items()
        }

    def _inject_global(self, input_ids, embeds, tables, history_item_ids):
        embeds = embeds.clone()
        hyperedge_token_id = self.config.hyperedge_token_id
        if hyperedge_token_id is None:
            raise RuntimeError("Missing hyperedge token ID in CRS configuration.")
        for view, table in tables.items():
            start_id = self.config.graph_start_token_ids.get(view)
            end_id = self.config.graph_end_token_ids.get(view)
            if start_id is None or end_id is None:
                raise RuntimeError(f"Missing graph token IDs for view '{view}'.")
            positions = []
            for row in range(input_ids.size(0)):
                ids = input_ids[row].tolist()
                try:
                    start = ids.index(start_id)
                    end = ids.index(end_id, start)
                except ValueError:
                    continue
                positions.extend(
                    (row, column)
                    for column in range(start + 1, end)
                    if ids[column] == hyperedge_token_id
                )
            item_ids = [item for row in history_item_ids.get(view, []) for item in row]
            if len(positions) != len(item_ids):
                raise ValueError(
                    f"{view} hyperedge placeholders ({len(positions)}) do not match "
                    f"history item IDs ({len(item_ids)})."
                )
            if positions:
                position_tensor = torch.tensor(positions, device=embeds.device)
                id_tensor = torch.tensor(
                    item_ids, device=table.device, dtype=torch.long
                )
                values = table.index_select(0, id_tensor).to(
                    device=embeds.device, dtype=embeds.dtype
                )
                embeds[position_tensor[:, 0], position_tensor[:, 1]] = values
        return embeds

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
            node_features = getattr(self, f"{feature_view}_feature_table").index_select(
                0, graph["node_ids"]
            )
            encoded = self.hypergraph_encoders[view](
                node_features, graph["hyperedge_index"], int(graph["edge_ptr"][-1])
            )
            projected_graph = self.hypergraph_projectors[view](encoded)
            projected = self.graph_moe(
                view=view,
                features=torch.cat(
                    (projected_graph.node_features, projected_graph.hyperedge_features)
                ),
            )
            node_count = graph["node_positions"].size(0)
            embeds[graph["node_positions"][:, 0], graph["node_positions"][:, 1]] = (
                projected[:node_count].to(embeds.dtype)
            )
            embeds[
                graph["hyperedge_positions"][:, 0], graph["hyperedge_positions"][:, 1]
            ] = projected[node_count:].to(embeds.dtype)
        return embeds

    def encode(
        self,
        input_ids,
        attention_mask=None,
        hypergraphs=None,
        global_hypergraphs=None,
        history_item_ids=None,
        **kwargs,
    ):
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("inputs_embeds", None)
        cache = kwargs.get("past_key_values")
        cached_step = cache is not None and cache.get_seq_length() > 0
        embeds = (
            self.get_input_embeddings()(input_ids)
            if cached_step
            else self._inject(input_ids, hypergraphs or {})
        )
        if not cached_step and self.config.global_hypergraph:
            if not self._global_hypergraph_outputs:
                self.cache_global_hypergraph_towers(global_hypergraphs or {})
            embeds = self._inject_global(
                input_ids,
                embeds,
                self.project_global_hyperedges(),
                history_item_ids or {},
            )
        if self.config.backbone_config.model_type == "qwen2_5_omni_thinker":
            kwargs["input_ids"] = input_ids

        # embeds = embeds[..., :1024, :]
        # if attention_mask is not None:
        #     attention_mask = attention_mask[:, :1024]
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
        self.recommendation_head = (
            None
            if config.global_hypergraph
            else RecommendationHead(_hidden_size(config.backbone_config), config)
        )
        self.global_recommendation_head = (
            GlobalHyperedgeMixtureHead(
                input_dim=_hidden_size(config.backbone_config),
                output_dim=config.recommendation_hidden_dim,
                views=tuple(config.views),
                temperature=config.recommendation_temperature,
            )
            if config.global_hypergraph
            else None
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pooling_mask=None,
        rec_labels=None,
        hypergraphs=None,
        global_hypergraphs=None,
        history_item_ids=None,
        **kwargs,
    ):
        output = self.encode(
            input_ids,
            attention_mask,
            hypergraphs,
            global_hypergraphs=global_hypergraphs,
            history_item_ids=history_item_ids,
            **kwargs,
        )
        hidden = output.hidden_states[-1]
        pooled = (hidden * pooling_mask.to(hidden.dtype).unsqueeze(-1)).sum(dim=1)
        pooled = pooled / pooling_mask.sum(dim=1, keepdim=True).to(hidden.dtype)
        if self.global_recommendation_head is not None:
            scores, _ = self.global_recommendation_head(
                pooled, self.project_global_hyperedges()
            )
            loss = F.nll_loss(scores, rec_labels) if rec_labels is not None else None
        else:
            assert self.recommendation_head is not None
            scores = self.recommendation_head(pooled, self._feature_tables())
            loss = (
                F.cross_entropy(scores, rec_labels) if rec_labels is not None else None
            )
        return SequenceClassifierOutput(loss=loss, logits=scores)


class HoCRSConversationModel(HoCRSBaseModel):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        hypergraphs=None,
        global_hypergraphs=None,
        history_item_ids=None,
        **kwargs,
    ):
        output = self.encode(
            input_ids,
            attention_mask,
            hypergraphs,
            global_hypergraphs=global_hypergraphs,
            history_item_ids=history_item_ids,
            labels=labels,
            **kwargs,
        )
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
    "GlobalHyperedgeMixtureHead",
    "hypergraph_propagate",
    "load_backbone",
]

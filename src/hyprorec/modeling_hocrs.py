"""Transformer-compatible HoCRS model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig


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
    """ Get the hidden size of the causal LM from its config. """
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if (
        text_config is not None
        and getattr(text_config, "hidden_size", None) is not None
    ):
        return int(text_config.hidden_size)
    raise ValueError("Cannot determine the causal LM hidden size from its config.")


def _cache_has_content(past_key_values: Any | None) -> bool:
    """ Check if the cache has content. """
    if past_key_values is None:
        return False
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if get_seq_length is not None:
        return int(get_seq_length()) > 0
    try:
        return int(past_key_values[0][0].shape[-2]) > 0
    except (IndexError, TypeError, AttributeError):
        return True


@dataclass
class HypergraphEncoderOutput:
    node_features: torch.Tensor
    hyperedge_features: torch.Tensor


@dataclass
class HoCRSOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    rec_scores: torch.Tensor | None = None
    rec_loss: torch.Tensor | None = None
    conv_loss: torch.Tensor | None = None
    moe_usage: torch.Tensor | None = None
    moe_router_entropy: torch.Tensor | None = None
    moe_view_usage: torch.Tensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


def hypergraph_propagate(
    node_features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    num_hyperedges: int,
) -> HypergraphEncoderOutput:
    """Apply symmetric HGNN node-edge-node propagation to a packed graph."""

    node_index, edge_index = hyperedge_index
    dtype = node_features.dtype
    node_degree = torch.bincount(
        node_index,
        minlength=node_features.size(0),
    ).to(dtype=dtype)
    edge_degree = torch.bincount(edge_index, minlength=num_hyperedges).to(dtype=dtype)
    node_scale = node_degree.clamp_min(1).pow(-0.5)
    edge_scale = edge_degree.clamp_min(1).reciprocal()

    normalized_nodes = node_scale.unsqueeze(-1) * node_features
    hyperedge_features = node_features.new_zeros(
        (num_hyperedges, node_features.size(-1))
    )
    hyperedge_features.index_add_(0, edge_index, normalized_nodes[node_index])
    hyperedge_features = edge_scale.unsqueeze(-1) * hyperedge_features

    propagated_nodes = node_features.new_zeros(node_features.shape)
    propagated_nodes.index_add_(0, node_index, hyperedge_features[edge_index])
    propagated_nodes = node_scale.unsqueeze(-1) * propagated_nodes
    return HypergraphEncoderOutput(propagated_nodes, hyperedge_features)


def _average_hyperedges(
    node_features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    num_hyperedges: int,
) -> HypergraphEncoderOutput:

    node_index, edge_index = hyperedge_index
    hyperedge_features = node_features.new_zeros(
        (num_hyperedges, node_features.size(-1))
    )
    hyperedge_features.index_add_(0, edge_index, node_features[node_index])
    edge_degree = torch.bincount(edge_index, minlength=num_hyperedges).to(
        dtype=node_features.dtype
    )
    hyperedge_features = hyperedge_features / edge_degree.clamp_min(1).unsqueeze(-1)
    return HypergraphEncoderOutput(node_features, hyperedge_features)


class HypergraphEncoder(nn.Module):
    """ Residual HGCN used independently by each graph view. """

    def __init__(self, config: HoCRSHypergraphConfig) -> None:
        super().__init__()
        dimensions = (
            [config.input_dim, config.output_dim]
            if config.num_layers == 1
            else [
                config.input_dim,
                *([config.hidden_dim] * (config.num_layers - 1)),
                config.output_dim,
            ]
        )
        self.layers = nn.ModuleList(
            nn.Linear(left, right, bias=config.use_bias)
            for left, right in zip(dimensions[:-1], dimensions[1:])
        )
        self.norms = nn.ModuleList(nn.LayerNorm(size) for size in dimensions[1:])
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_hyperedges: int,
    ) -> HypergraphEncoderOutput:
        output = None
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            transformed = layer(node_features)
            if index + 1 < len(self.layers):
                transformed = self.dropout(F.gelu(transformed))
            transformed = norm(transformed)
            output = hypergraph_propagate(
                transformed,
                hyperedge_index,
                num_hyperedges,
            )
            node_features = transformed + output.node_features
        assert output is not None
        return HypergraphEncoderOutput(node_features, output.hyperedge_features)


class HypergraphProjector(nn.Module):
    """ Map graph node and hyperedge states into the LM embedding space. """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.node_projector = nn.Linear(input_dim, output_dim)
        self.hyperedge_projector = nn.Linear(input_dim, output_dim)

    def forward(self, output: HypergraphEncoderOutput) -> HypergraphEncoderOutput:
        return HypergraphEncoderOutput(
            self.node_projector(output.node_features),
            self.hyperedge_projector(output.hyperedge_features),
        )


class GraphTokenMoE(nn.Module):
    """Dense token-level experts that add a residual to projected graph tokens."""

    def __init__(
        self,
        hidden_size: int,
        views: tuple[str, ...],
        num_experts: int,
        expert_hidden_size: int,
        router_temperature: float,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.router_temperature = router_temperature
        self.view_to_index = {view: index for index, view in enumerate(views)}
        self.view_embeddings = nn.Parameter(torch.empty(len(views), hidden_size))
        self.router = nn.Linear(hidden_size * 2, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, expert_hidden_size),
                    nn.GELU(),
                    nn.Linear(expert_hidden_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.residual_scales = nn.ParameterDict(
            {
                view: nn.Parameter(torch.tensor(residual_scale_init))
                for view in views
            }
        )
        nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)

    def reset_output_layers(self) -> None:
        """Start from the existing projected graph path before learning deltas."""

        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        view: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("Graph-token MoE expects a [tokens, hidden] tensor.")
        view_index = self.view_to_index[view]
        view_embedding = self.view_embeddings[view_index].expand(features.size(0), -1)
        router_input = torch.cat((features, view_embedding.to(features.dtype)), dim=-1)
        router_logits = self.router(router_input.float())
        weights = F.softmax(router_logits / self.router_temperature, dim=-1)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        delta = (weights.to(expert_outputs.dtype).unsqueeze(-1) * expert_outputs).sum(dim=1)
        output = features + self.residual_scales[view].to(features.dtype) * delta
        return output, weights, delta


class HoCRSRecommendationHead(nn.Module):
    """Full-catalog cosine head with ID or multimodal semantic items."""

    def __init__(self, input_dim: int, config: HoCRSConfig) -> None:
        super().__init__()
        self.user_projector = nn.Sequential(
            nn.Linear(input_dim, config.recommendation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.recommendation_dropout),
            nn.Linear(config.recommendation_hidden_dim, config.item_dim),
        )
        self.item_table_mode = config.item_table_mode
        self.semantic_views = tuple(config.views)
        if self.item_table_mode == "id":
            self.item_table = nn.Embedding(config.num_items, config.item_dim)
            self.semantic_projectors = nn.ModuleDict()
            self.item_residual = None
        else:
            self.item_table = None
            self.semantic_projectors = nn.ModuleDict(
                {
                    view: nn.Linear(
                        config.get_hypergraph_config(view).input_dim,
                        config.item_dim,
                        bias=False,
                    )
                    for view in self.semantic_views
                }
            )
            self.item_residual = nn.Embedding(config.num_items, config.item_dim)
        self.register_buffer(
            "item_frequency_gate",
            torch.zeros(config.num_items),
            persistent=True,
        )
        self.temperature = config.recommendation_temperature

    def reset_semantic_residual(self) -> None:
        if self.item_residual is not None:
            nn.init.zeros_(self.item_residual.weight)

    def set_item_frequencies(self, frequencies: torch.Tensor) -> None:
        if frequencies.shape != self.item_frequency_gate.shape:
            raise ValueError(
                f"Item frequencies have shape {tuple(frequencies.shape)}, "
                f"expected {tuple(self.item_frequency_gate.shape)}."
            )
        frequencies = frequencies.to(
            device=self.item_frequency_gate.device,
            dtype=self.item_frequency_gate.dtype,
        )
        maximum = frequencies.max()
        if maximum > 0:
            gate = torch.log1p(frequencies) / torch.log1p(maximum)
        else:
            gate = torch.zeros_like(frequencies)
        self.item_frequency_gate.copy_(gate)

    def item_embeddings(
        self,
        feature_tables: Mapping[str, torch.Tensor],
        semantic_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.item_table is not None:
            return F.normalize(self.item_table.weight, dim=-1)

        semantic = (
            semantic_table
            if semantic_table is not None
            else self.semantic_embeddings(feature_tables)
        )
        assert self.item_residual is not None
        residual = self.item_residual.weight * self.item_frequency_gate.unsqueeze(-1)
        return F.normalize(semantic + residual, dim=-1)

    def semantic_embeddings(
        self,
        feature_tables: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.semantic_projectors:
            raise ValueError("Semantic embeddings require semantic_hybrid mode.")
        semantic = None
        for view, projector in self.semantic_projectors.items():
            table = feature_tables[view]
            valid = table.abs().sum(dim=-1, keepdim=True) > 0
            projected = F.normalize(projector(table), dim=-1) * valid
            semantic = projected if semantic is None else semantic + projected
        if semantic is None:
            raise ValueError("semantic_hybrid requires modality feature tables.")
        return F.normalize(semantic, dim=-1)

    def semantic_node_features(
        self,
        view: str,
        node_ids: torch.Tensor,
        semantic_table: torch.Tensor,
    ) -> torch.Tensor:
        projector = self.semantic_projectors[view]
        shared_nodes = semantic_table.index_select(0, node_ids)
        return F.linear(shared_nodes, projector.weight.transpose(0, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_tables: Mapping[str, torch.Tensor],
        semantic_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        users = F.normalize(self.user_projector(hidden_states), dim=-1)
        items = self.item_embeddings(feature_tables, semantic_table)
        return users @ items.t() / self.temperature


class HoCRSModel(PreTrainedModel, GenerationMixin):
    """ End-to-end hypergraph prompting model for conversational recommendation. """

    config_class = HoCRSConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    accepts_loss_kwargs = False

    def __init__(self, config: HoCRSConfig, backbone: nn.Module | None = None) -> None:
        super().__init__(config)
        self.backbone = backbone or _backbone_from_config(config.backbone_config)
        lm_hidden_size = _hidden_size(config.backbone_config)

        self.hypergraph_encoders = nn.ModuleDict()
        self.hypergraph_projectors = nn.ModuleDict()
        for view in config.views:
            graph_config = config.get_hypergraph_config(view)
            if config.use_hypergraph_encoder:
                self.hypergraph_encoders[view] = HypergraphEncoder(graph_config)
            projector_input_dim = (
                graph_config.output_dim
                if config.use_hypergraph_encoder
                else graph_config.input_dim
            )
            self.hypergraph_projectors[view] = HypergraphProjector(projector_input_dim, lm_hidden_size)

            self.register_buffer(
                f"{view}_feature_table",
                torch.zeros(config.num_items, graph_config.input_dim),
                persistent=True,
            )

        self.moe = (
            GraphTokenMoE(
                hidden_size=lm_hidden_size,
                views=tuple(config.views),
                num_experts=config.moe_num_experts,
                expert_hidden_size=config.moe_hidden_dim,
                router_temperature=config.moe_router_temperature,
                residual_scale_init=config.moe_residual_scale_init,
            )
            if config.use_moe and config.views
            else None
        )
        self.recommendation_head = HoCRSRecommendationHead(lm_hidden_size, config)
        self.context_residual_projector = (
            nn.Linear(lm_hidden_size, lm_hidden_size, bias=False)
            if config.use_context_token
            else None
        )
        self.context_token_embedding = (
            nn.Embedding(1, lm_hidden_size) if config.use_context_token else None
        )
        self.soft_prompt_embeddings = nn.Embedding(
            config.num_soft_prompt_tokens,
            lm_hidden_size,
        ) if config.num_soft_prompt_tokens > 0 else None
        self.special_token_embeddings = (
            nn.Embedding(len(config.trainable_special_token_ids), lm_hidden_size)
            if config.train_special_tokens and config.trainable_special_token_ids
            else None
        )
        self._no_split_modules = list(
            dict.fromkeys(
                [
                    *getattr(self.backbone, "_no_split_modules", []),
                    "HypergraphEncoder",
                    "GraphTokenMoE",
                    "HoCRSRecommendationHead",
                ]
            )
        )
        self.post_init()
        self.recommendation_head.reset_semantic_residual()
        if self.context_residual_projector is not None:
            nn.init.zeros_(self.context_residual_projector.weight)
        if self.moe is not None:
            self.moe.reset_output_layers()
        self._initialize_special_token_embeddings()
        if self.context_token_embedding is not None:
            if config.context_token_id is None:
                raise ValueError("context_token_id is required when use_context_token is enabled.")
            with torch.no_grad():
                self.context_token_embedding.weight.copy_(
                    self.get_input_embeddings().weight[config.context_token_id].unsqueeze(0)
                )
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)

    def _initialize_special_token_embeddings(self) -> None:
        if self.special_token_embeddings is None: return
        source_embeddings = self.get_input_embeddings().weight
        if source_embeddings.device.type == "meta": return
        token_ids = torch.tensor(
            self.config.trainable_special_token_ids,
            dtype=torch.long,
            device=source_embeddings.device,
        )
        if int(token_ids.max()) >= self.get_input_embeddings().num_embeddings:
            raise ValueError("Resize the backbone token embeddings before creating HoCRS.")
        with torch.no_grad():
            source = source_embeddings.index_select(0, token_ids)
            self.special_token_embeddings.weight.copy_(source)

    def initialize_feature_tables(
        self,
        feature_tables: Mapping[str, torch.Tensor],
        item_frequencies: torch.Tensor | None = None,
    ) -> None:
        missing = set(self.config.views) - set(feature_tables)
        assert not missing, f"Missing feature tables: {sorted(missing)}"
        with torch.no_grad():

            # Initialize modality-based hypergraph node init features
            for view in self.config.views:
                target = getattr(self, f"{view}_feature_table")
                source = feature_tables[view].to(dtype=target.dtype, device=target.device)
                if source.shape != target.shape:
                    raise ValueError(
                        f"{view} feature table has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(source)
            if item_frequencies is not None:
                self.recommendation_head.set_item_frequencies(item_frequencies)

    def load_grounding_checkpoint(self, path: str) -> None:
        checkpoint = self._load_pretrained_state_dict(path)
        for view, encoder in self.hypergraph_encoders.items():
            prefix = f"hypergraph_encoders.{view}."
            view_state = {
                name.removeprefix(prefix): value
                for name, value in checkpoint.items()
                if name.startswith(prefix)
            }
            assert view_state, f"Grounding checkpoint has no '{view}' encoder."
            encoder.load_state_dict(view_state)
            encoder.requires_grad_(False)
            projector_prefix = f"hypergraph_projectors.{view}."
            projector_state = {
                name.removeprefix(projector_prefix): value
                for name, value in checkpoint.items()
                if name.startswith(projector_prefix)
            }
            if projector_state:
                self.hypergraph_projectors[view].load_state_dict(projector_state)
        self.config.grounding_checkpoint_path = path

    @staticmethod
    def _load_pretrained_state_dict(path: str) -> dict[str, torch.Tensor]:
        directory = Path(path)
        safetensor_path = directory / "model.safetensors"
        if safetensor_path.exists():
            from safetensors.torch import load_file
            return load_file(str(safetensor_path), device="cpu")
        return torch.load(directory / "pytorch_model.bin", map_location="cpu")

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module | None:
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_output_embeddings(value)

    def train(self, mode: bool = True) -> HoCRSModel:
        super().train(mode)
        self.backbone.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def _inject_special_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        result = inputs_embeds.clone()

        if self.context_token_embedding is not None:
            result[input_ids == self.config.context_token_id] = (
                self.context_token_embedding.weight[0].to(result.dtype)
            )

        # Inject special token embeddings if they are trainable.
        if self.special_token_embeddings is not None:
            embeddings = self.special_token_embeddings.weight.to(result.dtype)
            for index, token_id in enumerate(self.config.trainable_special_token_ids):
                result[input_ids == token_id] = embeddings[index]

        # Inject soft prompt embeddings if they are trainable.
        if self.config.num_soft_prompt_tokens > 0:
            assert self.soft_prompt_embeddings is not None

            positions = torch.nonzero(input_ids == self.config.soft_prompt_token_id, as_tuple=False)
            expected = input_ids.size(0) * self.config.num_soft_prompt_tokens
            if positions.size(0) != expected:
                raise ValueError(f"Expected {expected} soft prompt positions, found {positions.size(0)}.")
            prompt_ids = torch.arange(
                self.config.num_soft_prompt_tokens,
                device=input_ids.device,
            ).repeat(input_ids.size(0))  # (B * Ns, )
            result[positions[:, 0], positions[:, 1]] = self.soft_prompt_embeddings(prompt_ids).to(result.dtype)

        return result

    def _inject_hypergraphs(
        self,
        inputs_embeds: torch.Tensor,
        hypergraphs: Mapping[str, Mapping[str, torch.Tensor]] | None,
        semantic_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        result = inputs_embeds.clone()
        hypergraphs = hypergraphs or {}
        moe_weights: list[torch.Tensor] = []
        moe_view_weights: dict[str, torch.Tensor] = {}
        for view in self.config.views:
            if view not in hypergraphs:
                continue

            # Build node and hyperedge features for projection.
            graph = hypergraphs[view]
            node_ids = graph["node_ids"]
            if semantic_table is None:
                feature_table = getattr(self, f"{view}_feature_table")
                source_features = feature_table.index_select(0, node_ids)
            else:
                source_features = self.recommendation_head.semantic_node_features(
                    view,
                    node_ids,
                    semantic_table,
                )
            node_features = source_features
            num_hyperedges = int(graph["edge_ptr"][-1].item())
            if self.config.use_hypergraph_encoder:
                encoded = self.hypergraph_encoders[view](
                    node_features,
                    graph["hyperedge_index"],
                    num_hyperedges,
                )
            else:
                encoded = _average_hyperedges(
                    node_features,
                    graph["hyperedge_index"],
                    num_hyperedges,
                )
            projected = self.hypergraph_projectors[view](encoded)

            # Inject hypergraph features.
            node_positions = graph["node_positions"]
            hyperedge_positions = graph["hyperedge_positions"]
            if node_positions.size(0) != projected.node_features.size(0):
                raise ValueError(f"{view} node positions do not match encoder output.")
            if hyperedge_positions.size(0) != projected.hyperedge_features.size(0):
                raise ValueError(f"{view} hyperedge positions do not match encoder output.")
            if self.moe is not None:
                graph_features = torch.cat(
                    (projected.node_features, projected.hyperedge_features), dim=0
                )
                graph_features, weights, _ = self.moe(graph_features, view)
                node_count = projected.node_features.size(0)
                projected_nodes = graph_features[:node_count]
                projected_edges = graph_features[node_count:]
                moe_weights.append(weights)
                moe_view_weights[view] = weights.mean(dim=0)
            else:
                projected_nodes = projected.node_features
                projected_edges = projected.hyperedge_features
            result[node_positions[:, 0], node_positions[:, 1]] = projected_nodes.to(result.dtype)
            result[hyperedge_positions[:, 0], hyperedge_positions[:, 1]] = projected_edges.to(result.dtype)

        zero = result.sum() * 0.0
        if moe_weights:
            all_weights = torch.cat(moe_weights, dim=0)
            moe_usage = all_weights.mean(dim=0)
            moe_router_entropy = (
                -(all_weights.clamp_min(1e-8) * all_weights.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
            )
            moe_view_usage = torch.stack(
                [
                    moe_view_weights.get(
                        view,
                        zero.new_zeros(self.config.moe_num_experts),
                    )
                    for view in self.config.views
                ],
                dim=0,
            )
        else:
            expert_count = self.config.moe_num_experts if self.moe is not None else 0
            moe_usage = zero.new_zeros(expert_count)
            moe_router_entropy = zero
            moe_view_usage = zero.new_zeros((len(self.config.views), expert_count))
        return result, {
            "moe_usage": moe_usage,
            "moe_router_entropy": moe_router_entropy,
            "moe_view_usage": moe_view_usage,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        rec_labels: torch.Tensor | None = None,
        hypergraphs: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> HoCRSOutput:
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("inputs_embeds", None)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        is_cached_step = _cache_has_content(past_key_values)
        zero = inputs_embeds.sum() * 0.0
        expert_count = self.config.moe_num_experts if self.moe is not None else 0
        moe_diagnostics = {
            "moe_usage": zero.new_zeros(expert_count),
            "moe_router_entropy": zero,
            "moe_view_usage": zero.new_zeros((len(self.config.views), expert_count)),
        }
        semantic_table = None
        if not is_cached_step:
            inputs_embeds = self._inject_special_embeddings(input_ids, inputs_embeds)
            if self.config.views:
                if self.config.use_semantic_hypergraph_nodes:
                    feature_tables = {
                        view: getattr(self, f"{view}_feature_table")
                        for view in self.recommendation_head.semantic_views
                    }
                    semantic_table = self.recommendation_head.semantic_embeddings(
                        feature_tables
                    )
                inputs_embeds, moe_diagnostics = self._inject_hypergraphs(
                    inputs_embeds,
                    hypergraphs,
                    semantic_table,
                )
        if self.config.backbone_config.model_type == "qwen2_5_omni_thinker":
            kwargs["input_ids"] = input_ids

        backbone_output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
            **kwargs,
        )
        if is_cached_step:
            return HoCRSOutput(
                loss=backbone_output.loss,
                logits=backbone_output.logits,
                past_key_values=getattr(backbone_output, "past_key_values", None),
                hidden_states=backbone_output.hidden_states,
                attentions=getattr(backbone_output, "attentions", None),
            )
        assert self.config.rec_token_id is not None, "rec_token_id is missing from HoCRSConfig."
        rec_mask = input_ids == self.config.rec_token_id
        rec_hidden = backbone_output.hidden_states[-1][rec_mask]
        if self.context_residual_projector is not None:
            context_mask = input_ids == self.config.context_token_id
            context_hidden = backbone_output.hidden_states[-1][context_mask]
            rec_hidden = rec_hidden + self.context_residual_projector(context_hidden)
        semantic_tables = {
            view: getattr(self, f"{view}_feature_table")
            for view in self.recommendation_head.semantic_views
        }
        rec_scores = self.recommendation_head(
            rec_hidden,
            semantic_tables,
            semantic_table,
        )

        rec_loss = F.cross_entropy(rec_scores, rec_labels) if rec_labels is not None else None
        conv_loss = backbone_output.loss
        loss = None
        if rec_loss is not None and conv_loss is not None:
            loss = (
                self.config.beta * rec_loss
                + (1.0 - self.config.beta) * conv_loss
            )

        return HoCRSOutput(
            loss=loss,
            logits=backbone_output.logits,
            rec_scores=rec_scores,
            rec_loss=rec_loss,
            conv_loss=conv_loss,
            moe_usage=moe_diagnostics["moe_usage"],
            moe_router_entropy=moe_diagnostics["moe_router_entropy"],
            moe_view_usage=moe_diagnostics["moe_view_usage"],
            past_key_values=getattr(backbone_output, "past_key_values", None),
            hidden_states=backbone_output.hidden_states,
            attentions=getattr(backbone_output, "attentions", None),
        )


__all__ = [
    "HoCRSModel",
    "HoCRSOutput",
    "HypergraphEncoder",
    "HypergraphEncoderOutput",
    "HypergraphProjector",
    "GraphTokenMoE",
    "hypergraph_propagate",
]

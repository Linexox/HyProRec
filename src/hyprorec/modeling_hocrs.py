"""Transformer-compatible HoCRS model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel  # Modified: inspect backbone type before loading.
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput

from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig


# START: Load only Omni Thinker, while retaining the existing causal-LM path.
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
# END: Load only Omni Thinker, while retaining the existing causal-LM path.


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
    """Build direct-projection hyperedge features by averaging incident nodes."""

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


class HoCRSRecommendationHead(nn.Module):
    """ Full-catalog cosine recommendation head with an independent item table. """

    def __init__(self, input_dim: int, config: HoCRSConfig) -> None:
        super().__init__()
        self.user_projector = nn.Sequential(
            nn.Linear(input_dim, config.recommendation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.recommendation_dropout),
            nn.Linear(config.recommendation_hidden_dim, config.item_dim),
        )
        self.item_table = nn.Embedding(config.num_items, config.item_dim)
        self.temperature = config.recommendation_temperature

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        users = F.normalize(self.user_projector(hidden_states), dim=-1)
        items = F.normalize(self.item_table.weight, dim=-1)
        return users @ items.t() / self.temperature


class HoCRSModel(PreTrainedModel, GenerationMixin):
    """ End-to-end hypergraph prompting model for conversational recommendation. """

    config_class = HoCRSConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(self, config: HoCRSConfig, backbone: nn.Module | None = None) -> None:
        super().__init__(config)
        self.backbone = backbone or _backbone_from_config(config.backbone_config)  # Modified: reconstruct Thinker checkpoints.
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
            if view == "co":
                self.co_feature_table = nn.Parameter(torch.empty(config.num_items, graph_config.input_dim))
                nn.init.normal_(self.co_feature_table, mean=0.0, std=0.02)
            else:
                self.register_buffer(
                    f"{view}_feature_table",
                    torch.zeros(config.num_items, graph_config.input_dim),
                    persistent=True,
                )

        self.recommendation_head = HoCRSRecommendationHead(lm_hidden_size, config)
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
                    "HoCRSRecommendationHead",
                ]
            )
        )
        self.post_init()
        self._initialize_special_token_embeddings()
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)

    def _initialize_special_token_embeddings(self) -> None:
        if self.special_token_embeddings is None:
            return
        source_embeddings = self.get_input_embeddings().weight
        if source_embeddings.device.type == "meta":
            return
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
        item_table_init: torch.Tensor | None = None,
    ) -> None:
        semantic_views = set(self.config.views) - {"co"}
        missing = semantic_views - set(feature_tables)
        if missing:
            raise ValueError(f"Missing feature tables: {sorted(missing)}")
        with torch.no_grad():
            
            # Initialize modality-based hypergraph node init features
            for view in semantic_views:
                target = getattr(self, f"{view}_feature_table")
                source = feature_tables[view].to(dtype=target.dtype, device=target.device)
                if source.shape != target.shape:
                    raise ValueError(
                        f"{view} feature table has shape {tuple(source.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(source)
            
            # Initialize CO-view hypergraph node init features
            # if feature_tables["co"] is no provided, initialize it randomly
            if "co" in self.config.views and "co" in feature_tables:
                target = self.co_feature_table
                content = feature_tables["co"].to(dtype=target.dtype, device=target.device)
                if content.shape != target.shape:
                    raise ValueError(
                        f"co feature table has shape {tuple(content.shape)}, "
                        f"expected {tuple(target.shape)}."
                    )
                target.copy_(content)

            # Initialize item_table
            if item_table_init is not None:
                content = item_table_init.to(
                    dtype=self.recommendation_head.item_table.weight.dtype,
                    device=self.recommendation_head.item_table.weight.device,
                )
                if content.shape != self.recommendation_head.item_table.weight.shape:
                    raise ValueError("The Item Table content initialization has an incompatible shape.")
                self.recommendation_head.item_table.weight.copy_(content)

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
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def _inject_special_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        result = inputs_embeds.clone()

        # Inject special token embeddings if they are trainable.
        if self.special_token_embeddings is not None:
            embeddings = self.special_token_embeddings.weight.to(result.dtype)
            for index, token_id in enumerate(self.config.trainable_special_token_ids):
                result[input_ids == token_id] = embeddings[index]

        if self.config.num_soft_prompt_tokens > 0:
            if self.config.soft_prompt_token_id is None:
                raise ValueError("soft_prompt_token_id is missing from HoCRSConfig.")

            # Inject soft prompt embeddings.
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
        hypergraphs: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> torch.Tensor:
        result = inputs_embeds.clone()
        for view in self.config.views:
            if view not in hypergraphs:
                raise ValueError(f"Missing '{view}' hypergraph batch.")

            # Build node and hyperedge features for projection.
            graph = hypergraphs[view]
            node_ids = graph["node_ids"]
            feature_table = getattr(self, f"{view}_feature_table")
            node_features = feature_table.index_select(0, node_ids)
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
            result[node_positions[:, 0], node_positions[:, 1]] = (
                projected.node_features.to(result.dtype)
            )
            result[hyperedge_positions[:, 0], hyperedge_positions[:, 1]] = (
                projected.hyperedge_features.to(result.dtype)
            )

        return result

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
        inputs_embeds = self.get_input_embeddings()(input_ids)
        is_cached_step = _cache_has_content(past_key_values)
        if not is_cached_step:
            inputs_embeds = self._inject_special_embeddings(input_ids, inputs_embeds)
            if self.config.views:
                inputs_embeds = self._inject_hypergraphs(inputs_embeds, hypergraphs)
        # START: Thinker needs token IDs for its RoPE bookkeeping, even with injected embeddings.
        if self.config.backbone_config.model_type == "qwen2_5_omni_thinker":
            kwargs["input_ids"] = input_ids
        # END: Thinker needs token IDs for its RoPE bookkeeping, even with injected embeddings.
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
        if self.config.rec_token_id is None:
            raise ValueError("rec_token_id is missing from HoCRSConfig.")
        rec_mask = input_ids == self.config.rec_token_id
        counts = rec_mask.sum(dim=1)
        if not torch.all(counts == 1):
            raise ValueError("Every sample must contain exactly one recommendation token.")
        rec_hidden = backbone_output.hidden_states[-1][rec_mask]
        rec_scores = self.recommendation_head(rec_hidden)

        if rec_labels is not None:
            rec_loss = F.cross_entropy(rec_scores, rec_labels)
        else:
            rec_loss = None

        conv_loss = backbone_output.loss
        loss = None
        if rec_loss is not None and conv_loss is not None:
            loss = self.config.beta * rec_loss + (1.0 - self.config.beta) * conv_loss

        return HoCRSOutput(
            loss=loss,
            logits=backbone_output.logits,
            rec_scores=rec_scores,
            rec_loss=rec_loss,
            conv_loss=conv_loss,
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
    "hypergraph_propagate",
]

"""Typed experiment arguments for the independent HyProRec tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from accelerate import ParallelismConfig
from transformers import TrainingArguments

from .configuration_hocrs import ALL_GRAPH_VIEWS
from .constants import MODALITIES


@dataclass
class ModelArguments:
    backbone_name_or_path: str = "microsoft/DialoGPT-small"
    task: str = "recommendation"
    recommendation_head: str = "item_table"
    freeze_backbone: bool = True
    num_prompt_tokens: int = 20
    hypergraph_hidden_dim: int = 1024
    hypergraph_output_dim: int = 256
    hypergraph_num_layers: int = 3
    hypergraph_dropout: float = 0.0
    grounding_checkpoint_path: str | None = None
    grounding_ga_sa_weight: float = 1.0
    grounding_ga_sn_weight: float = 3.0
    grounding_sa_sn_weight: float = 3.0
    grounding_temperature: float = 0.07
    grounding_tokenizer_name_or_path: str = "sentence-transformers/all-mpnet-base-v2"
    item_table_view: str = "txt"
    co_feature_view: str = "txt"
    recommendation_hidden_dim: int = 2048
    recommendation_temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.task not in {"recommendation", "conversation"}:
            raise ValueError("task must be recommendation or conversation.")
        if self.recommendation_head not in {"item_table", "mlp"}:
            raise ValueError("recommendation_head must be item_table or mlp.")
        if self.item_table_view not in (*MODALITIES, "full"):
            raise ValueError("item_table_view must be a modality or full.")
        if self.co_feature_view not in MODALITIES:
            raise ValueError(f"co_feature_view must be one of {MODALITIES}.")
        if self.num_prompt_tokens != 20 or self.recommendation_hidden_dim < 1:
            raise ValueError(
                "num_prompt_tokens must be 20 and recommendation_hidden_dim positive."
            )


@dataclass
class DataArguments:
    dataset_path: str = "data/hocrs2_redial"
    hyperedge_table_path: str | None = None
    embeddings_dir_name: str = "embeddings"
    id_embeddings_path: str | None = None
    views: list[str] = field(default_factory=lambda: list(ALL_GRAPH_VIEWS))
    grounding_views: list[str] | None = None
    topk: int = 3
    khop: int = 2
    hyperedge_sampling: str = "strict"
    sample_repeat: int = 1
    max_history_tokens: int = 256
    global_hypergraph: bool = False

    def __post_init__(self) -> None:
        self.views = list(dict.fromkeys(self.views))
        if set(self.views) - set(ALL_GRAPH_VIEWS):
            raise ValueError(f"Unknown or duplicated graph views: {self.views}")
        if self.grounding_views is not None:
            allowed_grounding_views = (
                set(MODALITIES) | {f"co_{view}" for view in MODALITIES} | {"co"}
            )
            if (
                not self.grounding_views
                or len(set(self.grounding_views)) != len(self.grounding_views)
                or set(self.grounding_views) - allowed_grounding_views
            ):
                raise ValueError(
                    "grounding_views must contain unique modality or co-modality views."
                )
        if not 0 <= self.topk <= 10:
            raise ValueError("topk must be between 0 and 10.")
        if self.hyperedge_sampling not in {"strict", "random"}:
            raise ValueError("hyperedge_sampling must be strict or random.")
        if self.sample_repeat < 1 or not 0 < self.max_history_tokens <= 256:
            raise ValueError(
                "sample_repeat must be positive and max_history_tokens must be at most 256."
            )


@dataclass
class HoCRSTrainingArguments(TrainingArguments):
    output_dir: str = "outputs/redial/hocrs"
    remove_unused_columns: bool = False
    label_names: list[str] | None = field(
        default_factory=lambda: ["labels", "rec_labels"]
    )
    dp_replicate_size: int = 1
    dp_shard_size: int = 1

    def __post_init__(self) -> None:
        if self.dp_replicate_size < 1 or self.dp_shard_size < 1:
            raise ValueError("Data-parallel mesh dimensions must be positive.")
        if self.dp_shard_size == 1 and self.dp_replicate_size != 1:
            raise ValueError("Use ordinary DDP for replication-only training.")
        if self.dp_shard_size > 1:
            if self.parallelism_config is not None:
                raise ValueError("Set mesh dimensions or parallelism_config, not both.")
            self.fsdp = True
            self.fsdp_config = self.fsdp_config or {
                "version": 2,
                "reshard_after_forward": True,
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "state_dict_type": "FULL_STATE_DICT",
            }
            self.parallelism_config = ParallelismConfig(
                dp_replicate_size=self.dp_replicate_size,
                dp_shard_size=self.dp_shard_size,
            )
        super().__post_init__()


__all__ = ["DataArguments", "HoCRSTrainingArguments", "ModelArguments"]

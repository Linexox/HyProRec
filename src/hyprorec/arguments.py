"""Typed experiment arguments for HoCRS training."""

from __future__ import annotations

from dataclasses import dataclass, field

from accelerate import ParallelismConfig
from transformers import TrainingArguments

from .constants import GRAPH_VIEWS


@dataclass
class ModelArguments:
    backbone_name_or_path: str = "microsoft/DialoGPT-small"
    freeze_backbone: bool = True
    beta: float = 0.75
    num_soft_prompt_tokens: int = 10
    train_special_tokens: bool = False
    hypergraph_hidden_dim: int = 1024
    hypergraph_output_dim: int = 256
    hypergraph_num_layers: int = 3
    hypergraph_dropout: float = 0.0
    use_hypergraph_encoder: bool = True
    freeze_hypergraph_encoder: bool = False
    grounding_checkpoint_path: str | None = None
    recommendation_hidden_dim: int = 768
    recommendation_dropout: float = 0.0
    recommendation_temperature: float = 0.07


@dataclass
class DataArguments:
    dataset_path: str = "data/lhf-redial"
    hyperedge_table_path: str | None = None
    embeddings_dir_name: str = "embeddings"
    # START: Load the offline content initialization instead of fusing at train time.
    content_table_path: str = "data/lhf-redial/embeddings/content_full.pt"
    # END: Load the offline content initialization instead of fusing at train time.
    views: list[str] = field(default_factory=lambda: list(GRAPH_VIEWS))
    topk: int = 3
    khop: int = 2
    max_length: int | None = None

    def __post_init__(self) -> None:
        self.views = list(dict.fromkeys(self.views))
        unknown_views = set(self.views) - set(GRAPH_VIEWS)
        if unknown_views:
            raise ValueError(f"Unknown graph views: {sorted(unknown_views)}")


@dataclass
class GroundingArguments:
    """Grounding-specific data, encoder, and objective settings."""

    output_dir: str = "outputs/redial/v2/grounding"
    dataset_path: str = "data/lhf-redial"
    hyperedge_table_path: str | None = None
    embeddings_dir_name: str = "embeddings"
    modalities: list[str] = field(default_factory=lambda: list(GRAPH_VIEWS))
    topk: int = 3
    khop: int = 2
    validation_ratio: float = 0.1
    source_text_model: str = "sentence-transformers/all-mpnet-base-v2"
    source_image_model: str = "google/vit-base-patch16-224-in21k"
    source_audio_model: str = "facebook/wav2vec2-base"
    source_video_model: str = "MCG-NJU/videomae-base"
    source_hidden_dim: int = 256
    hypergraph_hidden_dim: int = 1024
    hypergraph_output_dim: int = 256
    hypergraph_num_layers: int = 3
    hypergraph_dropout: float = 0.0
    temperature: float = 0.07
    lambda_node: float = 1.0
    lambda_edge: float = 1.0
    lambda_member: float = 1.0
    member_negative_ratio: int = 1
    seed: int = 42
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        self.modalities = list(dict.fromkeys(self.modalities))
        unknown = set(self.modalities) - set(GRAPH_VIEWS)
        if unknown:
            raise ValueError(f"Unknown grounding views: {sorted(unknown)}")
        if not self.modalities:
            raise ValueError("At least one grounding view is required.")
        if self.topk < 1 or self.khop < 1:
            raise ValueError("topk and khop must be positive.")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in (0, 1).")
        if self.member_negative_ratio < 1:
            raise ValueError("member_negative_ratio must be positive.")


@dataclass
class HoCRSTrainingArguments(TrainingArguments):
    """TrainingArguments with explicit FSDP2/HSDP mesh dimensions."""

    output_dir: str = "outputs/redial/v1"
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
            raise ValueError(
                "Use ordinary DDP for replication-only training; "
                "HSDP requires dp_shard_size > 1."
            )
        if self.dp_shard_size > 1:
            if self.parallelism_config is not None:
                raise ValueError(
                    "Set dp_replicate_size/dp_shard_size or parallelism_config, not both."
                )
            self.fsdp = True
            if self.fsdp_config is None:
                self.fsdp_config = {
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


__all__ = [
    "DataArguments",
    "GroundingArguments",
    "HoCRSTrainingArguments",
    "ModelArguments",
]

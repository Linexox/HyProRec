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
    grounding_checkpoint_path: str | None = None
    use_source_projector: bool = False
    grounding_weight: float = 0.0
    grounding_ga_sa_weight: float = 1.0
    grounding_ga_sn_weight: float = 1.0
    grounding_sa_sn_weight: float = 0.0
    grounding_temperature: float = 0.07
    item_table_init: str = "random"
    train_item_table: bool = True
    recommendation_hidden_dim: int = 2048  # Modified: match HoCRS2 recommendation space.
    grounding_tokenizer_name_or_path: str = "sentence-transformers/all-mpnet-base-v2"  # Modified: raw-text Grounding tokenizer.
    recommendation_dropout: float = 0.0
    recommendation_temperature: float = 0.07

    def __post_init__(self) -> None:
        if self.item_table_init not in {"aligned_content", "random"}:
            raise ValueError("item_table_init must be 'aligned_content' or 'random'.")
        if self.grounding_weight < 0 or any(
            weight < 0
            for weight in (
                self.grounding_ga_sa_weight,
                self.grounding_ga_sn_weight,
                self.grounding_sa_sn_weight,
            )
        ):
            raise ValueError("Grounding weights must be non-negative.")
        if self.grounding_temperature <= 0:
            raise ValueError("grounding_temperature must be positive.")
        # START: The standalone Grounding stage has its own trainable source encoder.
        if (
            self.grounding_weight > 0
            and self.grounding_sa_sn_weight > 0
            and not self.use_source_projector
        ):
            raise ValueError(
                "grounding_sa_sn_weight has no trainable effect without "
                "use_source_projector."
            )
        # END: The standalone Grounding stage has its own trainable source encoder.


@dataclass
class DataArguments:
    dataset_path: str = "data/lhf-redial"
    hyperedge_table_path: str | None = None
    embeddings_dir_name: str = "embeddings"
    content_table_path: str = "data/lhf-redial/embeddings/content_full.pt"
    views: list[str] = field(default_factory=lambda: list(GRAPH_VIEWS))
    topk: int = 3
    khop: int = 2
    max_length: int = 1024
    max_history_tokens: int = 150
    max_response_tokens: int = 64

    def __post_init__(self) -> None:
        self.views = list(dict.fromkeys(self.views))
        unknown_views = set(self.views) - set(GRAPH_VIEWS)
        if unknown_views:
            raise ValueError(f"Unknown graph views: {sorted(unknown_views)}")


@dataclass
class AlignmentArguments:
    dataset_path: str = "data/lhf-redial"
    modalities: list[str] = field(default_factory=lambda: list(GRAPH_VIEWS[1:]))
    txt_model_name_or_path: str = "sentence-transformers/all-mpnet-base-v2"
    img_model_name_or_path: str = "google/vit-base-patch16-224-in21k"
    ado_model_name_or_path: str = "facebook/wav2vec2-base"
    vdo_model_name_or_path: str = "MCG-NJU/videomae-base"
    alignment_dim: int = 256
    temperature: float = 0.07
    validation_ratio: float = 0.1
    max_text_length: int = 128

    def __post_init__(self) -> None:
        modalities = tuple(GRAPH_VIEWS[1:])
        self.modalities = list(dict.fromkeys(self.modalities))
        unknown = set(self.modalities) - set(modalities)
        if unknown:
            raise ValueError(f"Unknown alignment modalities: {sorted(unknown)}")
        if len(self.modalities) < 2:
            raise ValueError("Alignment requires at least two modalities.")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in (0, 1).")


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
    "AlignmentArguments",
    "DataArguments",
    "HoCRSTrainingArguments",
    "ModelArguments",
]

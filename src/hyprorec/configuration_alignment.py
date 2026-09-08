"""Serializable configuration for multimodal item alignment."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from .constants import MODALITIES

DEFAULT_SOURCE_MODELS = {
    "txt": "sentence-transformers/all-mpnet-base-v2",
    "img": "google/vit-base-patch16-224-in21k",
    "ado": "facebook/wav2vec2-base",
    "vdo": "MCG-NJU/videomae-base",
}


class HoCRSAlignmentConfig(PretrainedConfig):
    """Source encoders and the common item embedding space used by v3."""

    model_type = "hocrs_alignment"

    def __init__(
        self,
        modalities: list[str] | tuple[str, ...] = MODALITIES,
        txt_model_name_or_path: str = DEFAULT_SOURCE_MODELS["txt"],
        img_model_name_or_path: str = DEFAULT_SOURCE_MODELS["img"],
        ado_model_name_or_path: str = DEFAULT_SOURCE_MODELS["ado"],
        vdo_model_name_or_path: str = DEFAULT_SOURCE_MODELS["vdo"],
        alignment_dim: int = 256,
        temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        modalities = tuple(dict.fromkeys(modalities))
        unknown = set(modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"Unknown alignment modalities: {sorted(unknown)}")
        if len(modalities) < 2:
            raise ValueError("Alignment requires at least two modalities.")
        if alignment_dim <= 0:
            raise ValueError("alignment_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.modalities = modalities
        self.txt_model_name_or_path = txt_model_name_or_path
        self.img_model_name_or_path = img_model_name_or_path
        self.ado_model_name_or_path = ado_model_name_or_path
        self.vdo_model_name_or_path = vdo_model_name_or_path
        self.alignment_dim = alignment_dim
        self.temperature = temperature

    def get_model_name_or_path(self, modality: str) -> str:
        if modality not in MODALITIES:
            raise KeyError(f"Unknown modality: {modality}")
        return getattr(self, f"{modality}_model_name_or_path")


__all__ = ["HoCRSAlignmentConfig"]

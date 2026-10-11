"""Backbone adapters used by the HyProRec CRS model."""

from .base import (
    BackboneAdapter,
    BackboneOutput,
    adapter_from_config,
    adapter_from_model,
    load_backbone_adapter,
)

__all__ = [
    "BackboneAdapter",
    "BackboneOutput",
    "adapter_from_config",
    "adapter_from_model",
    "load_backbone_adapter",
]

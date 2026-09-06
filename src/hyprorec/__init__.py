"""HyProRec public API."""

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from .modeling_hocrs import HoCRSModel, HoCRSOutput
from .processing_hocrs import HoCRSProcessor

AutoConfig.register(HoCRSConfig.model_type, HoCRSConfig, exist_ok=True)
AutoModelForCausalLM.register(HoCRSConfig, HoCRSModel, exist_ok=True)


__all__ = [
    "HoCRSConfig",
    "HoCRSHypergraphConfig",
    "HoCRSModel",
    "HoCRSOutput",
    "HoCRSProcessor",
]

"""HyProRec public API."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from .modeling_hocrs import GraphTokenMoE, HoCRSModel, HoCRSOutput
from .configuration_grounding import HoCRSGroundingConfig
from .modeling_grounding import HoCRSGroundingModel, HoCRSGroundingOutput
from .processing_hocrs import HoCRSProcessor

AutoConfig.register(HoCRSConfig.model_type, HoCRSConfig, exist_ok=True)
AutoModelForCausalLM.register(HoCRSConfig, HoCRSModel, exist_ok=True)
AutoConfig.register(HoCRSGroundingConfig.model_type, HoCRSGroundingConfig, exist_ok=True)
AutoModel.register(HoCRSGroundingConfig, HoCRSGroundingModel, exist_ok=True)


__all__ = [
    "HoCRSConfig",
    "HoCRSHypergraphConfig",
    "HoCRSModel",
    "HoCRSOutput",
    "GraphTokenMoE",
    "HoCRSProcessor",
    "HoCRSGroundingConfig",
    "HoCRSGroundingModel",
    "HoCRSGroundingOutput",
]

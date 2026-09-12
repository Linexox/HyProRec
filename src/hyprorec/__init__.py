"""HyProRec public API."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# START: Register the v3 alignment checkpoint with Transformers auto classes.
from .configuration_alignment import HoCRSAlignmentConfig
from .modeling_alignment import HoCRSAlignmentModel, HoCRSAlignmentOutput
# END: Register the v3 alignment checkpoint with Transformers auto classes.
from .configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from .modeling_hocrs import GraphTokenMoE, HoCRSModel, HoCRSOutput
from .configuration_grounding import HoCRSGroundingConfig
from .modeling_grounding import HoCRSGroundingModel, HoCRSGroundingOutput
from .processing_hocrs import HoCRSProcessor

AutoConfig.register(HoCRSConfig.model_type, HoCRSConfig, exist_ok=True)
# START: Enable standard AutoConfig/AutoModel loading for alignment checkpoints.
AutoConfig.register(
    HoCRSAlignmentConfig.model_type, HoCRSAlignmentConfig, exist_ok=True
)
AutoModel.register(HoCRSAlignmentConfig, HoCRSAlignmentModel, exist_ok=True)
# END: Enable standard AutoConfig/AutoModel loading for alignment checkpoints.
AutoModelForCausalLM.register(HoCRSConfig, HoCRSModel, exist_ok=True)
# START: Register the standalone Grounding checkpoint with Transformers.
AutoConfig.register(
    HoCRSGroundingConfig.model_type, HoCRSGroundingConfig, exist_ok=True
)
AutoModel.register(HoCRSGroundingConfig, HoCRSGroundingModel, exist_ok=True)
# END: Register the standalone Grounding checkpoint with Transformers.


__all__ = [
    # START: Expose the v3 alignment model through the package API.
    "HoCRSAlignmentConfig",
    "HoCRSAlignmentModel",
    "HoCRSAlignmentOutput",
    # END: Expose the v3 alignment model through the package API.
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

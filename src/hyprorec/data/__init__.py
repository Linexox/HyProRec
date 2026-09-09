"""Data structures and datasets used by HoCRS."""

from .batch import BatchData, HypergraphBatch, batch_hypergraphs

# Modified: expose raw alignment data components for the dedicated stage.
from .alignment import HoCRSAlignmentCollator, HoCRSAlignmentDataset, split_item_ids
from .hypergraph import HypergraphData, HypergraphTable
from .grounding import HoCRSGroundingCollator, HoCRSGroundingDataset
from .redial import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig

__all__ = [
    "BatchData",
    "HoCRSAlignmentCollator",  # Modified: export the alignment collator.
    "HoCRSAlignmentDataset",  # Modified: export the alignment dataset.
    "HoCRSDataCollator",
    "HoCRSGroundingCollator",
    "HoCRSGroundingDataset",
    "HoCRSDataset",
    "HoCRSDatasetConfig",
    "HypergraphBatch",
    "HypergraphData",
    "HypergraphTable",
    "batch_hypergraphs",
    "split_item_ids",  # Modified: export the deterministic catalogue split.
]

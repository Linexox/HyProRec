"""Shared names used by data preparation, processing, and modeling."""

MODALITIES = ("txt", "img", "ado", "vdo")
GRAPH_VIEWS = ("co", *MODALITIES)


__all__ = ["GRAPH_VIEWS", "MODALITIES"]

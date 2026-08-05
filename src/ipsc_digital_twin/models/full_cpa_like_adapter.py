"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_adapter import RegenAIPTForwardAdapter

FullCPALikeForwardAdapter = RegenAIPTForwardAdapter

__all__ = ["FullCPALikeForwardAdapter", "RegenAIPTForwardAdapter"]

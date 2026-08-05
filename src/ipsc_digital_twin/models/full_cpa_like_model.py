"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_model import (
    GradientReversalFunction,
    GradientReversalLayer,
    RegenAIPTNet,
)

FullCPALikeNet = RegenAIPTNet

__all__ = [
    "FullCPALikeNet",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "RegenAIPTNet",
]

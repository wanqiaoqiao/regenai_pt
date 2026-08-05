"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_data import (
    CategoryEncoder,
    RegenAIPTDataBundle,
    RegenAIPTDataset,
    RegenAIPTMappings,
    build_regenai_pt_dataloaders,
    prepare_round_specific_adata,
    require_torch,
)

FullCPALikeDataBundle = RegenAIPTDataBundle
FullCPALikeDataset = RegenAIPTDataset
FullCPALikeMappings = RegenAIPTMappings
build_full_cpa_like_dataloaders = build_regenai_pt_dataloaders

__all__ = [
    "CategoryEncoder",
    "FullCPALikeDataBundle",
    "FullCPALikeDataset",
    "FullCPALikeMappings",
    "RegenAIPTDataBundle",
    "RegenAIPTDataset",
    "RegenAIPTMappings",
    "build_full_cpa_like_dataloaders",
    "build_regenai_pt_dataloaders",
    "prepare_round_specific_adata",
    "require_torch",
]

"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_config import (
    RegenAIPTConfig,
    RegenAIPTValidationReport,
    validate_regenai_pt_adata,
)

FullCPALikeConfig = RegenAIPTConfig
FullCPALikeValidationReport = RegenAIPTValidationReport
validate_full_cpa_like_adata = validate_regenai_pt_adata

__all__ = [
    "FullCPALikeConfig",
    "FullCPALikeValidationReport",
    "RegenAIPTConfig",
    "RegenAIPTValidationReport",
    "validate_full_cpa_like_adata",
    "validate_regenai_pt_adata",
]

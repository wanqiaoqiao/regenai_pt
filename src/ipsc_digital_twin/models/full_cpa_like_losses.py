"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_losses import compute_regenai_pt_loss

compute_full_cpa_like_loss = compute_regenai_pt_loss

__all__ = ["compute_full_cpa_like_loss", "compute_regenai_pt_loss"]

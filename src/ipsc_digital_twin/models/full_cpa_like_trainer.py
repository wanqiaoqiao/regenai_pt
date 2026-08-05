"""Backward-compatible imports for the former full_cpa_like module name."""

from .regenai_pt_trainer import RegenAIPTTrainer

FullCPALikeTrainer = RegenAIPTTrainer

__all__ = ["FullCPALikeTrainer", "RegenAIPTTrainer"]

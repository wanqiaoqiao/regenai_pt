from .forward_interface import (
    CurrentStateProfile,
    ForwardTransitionModelInterface,
    PredictedState,
    TargetStateProfile,
)
from .regenai_pt_config import RegenAIPTConfig, RegenAIPTValidationReport, validate_regenai_pt_adata
from .regenai_pt_data import (
    CategoryEncoder,
    RegenAIPTDataBundle,
    RegenAIPTDataset,
    RegenAIPTMappings,
    build_regenai_pt_dataloaders,
    prepare_round_specific_adata,
    require_torch,
)
from .simulation import simulate_round1, simulate_round2, simulate_two_round_sequence

try:  # optional torch-backed losses
    from .regenai_pt_losses import compute_regenai_pt_loss
except ImportError:  # pragma: no cover
    compute_regenai_pt_loss = None  # type: ignore[assignment]

try:  # optional torch-backed trainer
    from .regenai_pt_trainer import RegenAIPTTrainer
except ImportError:  # pragma: no cover
    RegenAIPTTrainer = None  # type: ignore[misc,assignment]

try:  # optional torch-backed adapter
    from .regenai_pt_adapter import RegenAIPTForwardAdapter
except ImportError:  # pragma: no cover
    RegenAIPTForwardAdapter = None  # type: ignore[misc,assignment]

# Compatibility aliases for the pre-RegenAI-PT public API.
FullCPALikeConfig = RegenAIPTConfig
FullCPALikeDataBundle = RegenAIPTDataBundle
FullCPALikeDataset = RegenAIPTDataset
FullCPALikeForwardAdapter = RegenAIPTForwardAdapter
FullCPALikeMappings = RegenAIPTMappings
FullCPALikeTrainer = RegenAIPTTrainer
FullCPALikeValidationReport = RegenAIPTValidationReport
build_full_cpa_like_dataloaders = build_regenai_pt_dataloaders
compute_full_cpa_like_loss = compute_regenai_pt_loss
validate_full_cpa_like_adata = validate_regenai_pt_adata

__all__ = [
    'CategoryEncoder',
    'CurrentStateProfile',
    'ForwardTransitionModelInterface',
    'RegenAIPTConfig',
    'RegenAIPTDataBundle',
    'RegenAIPTDataset',
    'RegenAIPTForwardAdapter',
    'RegenAIPTMappings',
    'RegenAIPTTrainer',
    'RegenAIPTValidationReport',
    'PredictedState',
    'TargetStateProfile',
    'build_regenai_pt_dataloaders',
    'compute_regenai_pt_loss',
    'prepare_round_specific_adata',
    'require_torch',
    'simulate_round1',
    'simulate_round2',
    'simulate_two_round_sequence',
    'validate_regenai_pt_adata',
]

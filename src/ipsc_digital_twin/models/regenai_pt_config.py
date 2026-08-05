from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import anndata as ad
import numpy as np

_ALLOWED_INPUT_LAYERS = {"X", "raw_counts", "log_normalized"}
_ALLOWED_RECON_LOSSES = {"mse", "nb", "zinb"}
_ALLOWED_DEVICES = {"auto", "cpu", "cuda"}
_ALLOWED_TREATMENT_ROLES = {"round1_treatment", "round2_treatment"}
_MIN_RECOMMENDED_CELLS_PER_TREATMENT = 10


def _parse_component_like_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            import json

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        for separator in ("|", ";", "+", ","):
            if separator in stripped:
                return [part.strip() for part in stripped.split(separator) if part.strip()]
        return [stripped]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


@dataclass(slots=True)
class RegenAIPTConfig:
    model_type: str = "regenai_pt"
    input_layer: Literal["X", "raw_counts", "log_normalized"] = "X"
    n_latent: int = 32
    n_hidden: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    treatment_key: str = "treatment"
    dose_key: str | None = "dose"
    control_treatment: str = "control"
    covariate_keys: tuple[str, ...] = field(default_factory=tuple)
    round_key: str | None = "round"
    time_key: str | None = "time_point"
    batch_key: str | None = "batch"
    line_key: str | None = "iPSC_line"
    round1_treatment_key: str = "round1_treatment"
    round1_dose_key: str = "round1_dose"
    round2_treatment_key: str = "round2_treatment"
    round2_dose_key: str = "round2_dose"
    round1_components_key: str = "round1_components"
    round1_component_doses_key: str = "round1_doses"
    round1_component_dose_units_key: str = "round1_dose_units"
    round2_components_key: str = "round2_components"
    round2_component_doses_key: str = "round2_doses"
    round2_component_dose_units_key: str = "round2_dose_units"
    component_key: str = "treatment_components"
    component_dose_key: str = "treatment_component_doses"
    component_dose_unit_key: str = "treatment_component_dose_units"
    max_components: int = 4
    use_component_interactions: bool = True
    treatment_role_key: str = "treatment_role"
    treatment_sequence_key: str = "treatment_sequence"
    max_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    reconstruction_loss: Literal["mse", "nb", "zinb"] = "mse"
    adversarial_weight: float = 1.0
    covariate_adversarial_weight: float = 0.5
    perturbation_adversarial_weight: float = 0.5
    embedding_l2_weight: float = 1e-4
    dose_regularization_weight: float = 0.1
    random_seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self) -> None:
        if self.model_type not in {"regenai_pt", "full_cpa_like"}:
            raise ValueError(
                "RegenAIPTConfig.model_type must be 'regenai_pt' "
                "or the legacy alias 'full_cpa_like'"
            )
        self.model_type = "regenai_pt"
        if self.input_layer not in _ALLOWED_INPUT_LAYERS:
            raise ValueError(f"input_layer must be one of {sorted(_ALLOWED_INPUT_LAYERS)}")
        if self.reconstruction_loss not in _ALLOWED_RECON_LOSSES:
            raise ValueError(f"reconstruction_loss must be one of {sorted(_ALLOWED_RECON_LOSSES)}")
        if self.device not in _ALLOWED_DEVICES:
            raise ValueError(f"device must be one of {sorted(_ALLOWED_DEVICES)}")
        if self.n_latent <= 0 or self.n_hidden <= 0 or self.n_layers <= 0:
            raise ValueError("n_latent, n_hidden, and n_layers must be positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.max_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("max_epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.adversarial_weight < 0.0:
            raise ValueError("adversarial_weight must be non-negative")
        if self.covariate_adversarial_weight < 0.0:
            raise ValueError("covariate_adversarial_weight must be non-negative")
        if self.perturbation_adversarial_weight < 0.0:
            raise ValueError("perturbation_adversarial_weight must be non-negative")
        if self.embedding_l2_weight < 0.0:
            raise ValueError("embedding_l2_weight must be non-negative")
        if self.dose_regularization_weight < 0.0:
            raise ValueError("dose_regularization_weight must be non-negative")
        if not self.treatment_key:
            raise ValueError("treatment_key must be a non-empty string")
        if not self.control_treatment:
            raise ValueError("control_treatment must be a non-empty string")
        if self.max_components <= 0:
            raise ValueError("max_components must be positive")
        self.covariate_keys = tuple(key for key in self.covariate_keys if key)

@dataclass(slots=True)
class RegenAIPTValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_cells: int = 0
    n_treatments: int = 0
    treatment_counts: dict[str, int] = field(default_factory=dict)
    component_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

def _get_expression_matrix(adata: ad.AnnData, config: RegenAIPTConfig):
    if config.input_layer == "X":
        return adata.X
    return adata.layers.get(config.input_layer)


def _normalized_treatment_labels(obs: Any, config: RegenAIPTConfig) -> list[str]:
    if config.treatment_key in obs.columns:
        return obs[config.treatment_key].astype("string").fillna("").astype(str).tolist()
    if config.component_key in obs.columns:
        labels: list[str] = []
        for raw in obs[config.component_key].tolist():
            components = _parse_component_like_value(raw)
            labels.append("+".join(components) if components else "")
        return labels
    return []


def validate_regenai_pt_adata(adata: ad.AnnData, config: RegenAIPTConfig) -> RegenAIPTValidationReport:
    report = RegenAIPTValidationReport(n_cells=int(adata.n_obs))

    matrix = _get_expression_matrix(adata, config)
    if matrix is None:
        if config.input_layer == "X":
            report.errors.append("AnnData.X is required when input_layer='X'")
        else:
            report.errors.append(
                f"AnnData layer '{config.input_layer}' is required when input_layer='{config.input_layer}'"
            )

    obs = adata.obs
    required_obs_keys = list(config.covariate_keys)
    optional_required = [config.dose_key, config.round_key, config.time_key]
    if config.batch_key:
        required_obs_keys.append(config.batch_key)
    if config.line_key:
        required_obs_keys.append(config.line_key)
    required_obs_keys.extend(key for key in optional_required if key)

    has_treatment_labels = config.treatment_key in obs.columns
    has_components = config.component_key in obs.columns
    if not has_treatment_labels and not has_components:
        report.errors.append(
            f"Required adata.obs column missing: '{config.treatment_key}' or '{config.component_key}'"
        )

    for key in required_obs_keys:
        if key not in obs.columns:
            report.errors.append(f"Required adata.obs column missing: '{key}'")

    if config.treatment_role_key and config.treatment_role_key in obs.columns:
        roles = set(obs[config.treatment_role_key].astype(str).dropna().tolist())
        invalid = sorted(role for role in roles if role not in _ALLOWED_TREATMENT_ROLES)
        if invalid:
            report.errors.append(
                f"adata.obs['{config.treatment_role_key}'] contains invalid values: {', '.join(invalid)}"
            )

    if has_components:
        component_totals: dict[str, int] = {}
        empty_components = 0
        for raw in obs[config.component_key].tolist():
            components = _parse_component_like_value(raw)
            if not components:
                empty_components += 1
            if len(components) > config.max_components:
                report.errors.append(
                    f"Encountered treatment with {len(components)} components, exceeding max_components={config.max_components}"
                )
            for component in components:
                component_totals[component] = component_totals.get(component, 0) + 1
        report.component_counts = component_totals
        if empty_components:
            report.errors.append(f"adata.obs['{config.component_key}'] contains empty component sets")

    treatments = np.asarray(_normalized_treatment_labels(obs, config), dtype=object)
    if treatments.size:
        empty_mask = np.array([not str(value).strip() for value in treatments], dtype=bool)
        if bool(empty_mask.any()):
            report.errors.append(f"adata.obs treatment labels derived from '{config.treatment_key}' contain empty values")

        non_empty_treatments = [str(value) for value in treatments[~empty_mask].tolist()]
        if config.control_treatment not in set(non_empty_treatments):
            report.errors.append(
                f"Control treatment '{config.control_treatment}' was not found in derived treatment labels"
            )

        counts: dict[str, int] = {}
        for label in non_empty_treatments:
            counts[label] = counts.get(label, 0) + 1
        report.treatment_counts = counts
        report.n_treatments = len(counts)

        too_small = {
            treatment: count
            for treatment, count in report.treatment_counts.items()
            if count < _MIN_RECOMMENDED_CELLS_PER_TREATMENT
        }
        if too_small:
            msg = ", ".join(f"{treatment}={count}" for treatment, count in sorted(too_small.items()))
            report.warnings.append(
                "Some treatments have low cell counts for RegenAI-PT training; "
                f"recommended minimum is {_MIN_RECOMMENDED_CELLS_PER_TREATMENT} cells per treatment ({msg})"
            )

    if has_components and config.component_dose_key in obs.columns:
        for idx, (raw_components, raw_doses) in enumerate(zip(obs[config.component_key].tolist(), obs[config.component_dose_key].tolist(), strict=False)):
            components = _parse_component_like_value(raw_components)
            doses = _parse_component_like_value(raw_doses)
            if doses and len(doses) != len(components):
                report.errors.append(
                    f"Row {idx} has mismatched component/dose lengths in '{config.component_key}' and '{config.component_dose_key}'"
                )
                break

    if matrix is not None:
        shape = getattr(matrix, "shape", None)
        if shape is not None:
            if int(shape[0]) != int(adata.n_obs):
                report.errors.append("Expression matrix row count does not match adata.n_obs")
            if len(shape) < 2 or int(shape[1]) <= 0:
                report.errors.append("Expression matrix must have at least one feature/gene column")

    if config.reconstruction_loss in {"nb", "zinb"} and matrix is not None:
        matrix_array = np.asarray(matrix)
        if np.any(matrix_array < 0):
            report.errors.append(
                f"reconstruction_loss='{config.reconstruction_loss}' requires non-negative expression values"
            )

    return report


__all__ = [
    "RegenAIPTConfig",
    "RegenAIPTValidationReport",
    "validate_regenai_pt_adata",
]

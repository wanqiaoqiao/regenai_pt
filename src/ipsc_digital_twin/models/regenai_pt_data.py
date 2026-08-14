from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np

from .regenai_pt_config import (
    RegenAIPTConfig,
    _parse_component_like_value,
    validate_regenai_pt_adata,
)

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset

    _TORCH_IMPORT_ERROR: ImportError | None
else:
    try:  # pragma: no cover - behavior is exercised through require_torch
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - tested indirectly
        torch = None
        DataLoader = Any
        Dataset = object
        _TORCH_IMPORT_ERROR = exc
    else:  # pragma: no cover - import path depends on environment
        _TORCH_IMPORT_ERROR = None


_TORCH_ERROR_MESSAGE = (
    "PyTorch is required for RegenAI-PT dataset/dataloader utilities. "
    "Install torch to use the '--transition-model regenai_pt' backend."
)


@dataclass(slots=True)
class CategoryEncoder:
    categories_: list[str] = field(default_factory=list)
    category_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, values: list[Any]) -> CategoryEncoder:
        categories = sorted({str(value) for value in values})
        return cls(categories_=categories, category_to_id={category: idx for idx, category in enumerate(categories)})

    def transform(self, values: list[Any]) -> np.ndarray:
        missing = sorted({str(value) for value in values if str(value) not in self.category_to_id})
        if missing:
            raise KeyError(f"Unknown category values encountered: {', '.join(missing)}")
        return np.asarray([self.category_to_id[str(value)] for value in values], dtype=np.int64)

    def inverse_transform(self, ids: list[int] | np.ndarray) -> list[str]:
        return [self.categories_[int(idx)] for idx in ids]


@dataclass(slots=True)
class RegenAIPTMappings:
    treatment_to_id: dict[str, int]
    id_to_treatment: dict[int, str]
    component_to_id: dict[str, int]
    id_to_component: dict[int, str]
    covariate_to_id: dict[str, dict[str, int]]
    id_to_covariate: dict[str, dict[int, str]]
    gene_names: list[str]
    input_dim: int
    max_components: int


@dataclass(slots=True)
class RegenAIPTDataBundle:
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    train_loader: Any
    val_loader: Any
    test_loader: Any
    mappings: RegenAIPTMappings

def require_torch() -> None:
    if torch is None:
        raise ImportError(_TORCH_ERROR_MESSAGE) from _TORCH_IMPORT_ERROR


def _to_numpy(matrix: Any) -> np.ndarray:
    if matrix is None:
        raise ValueError("Expression matrix is missing")
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _get_expression_matrix(adata: ad.AnnData, config: RegenAIPTConfig) -> np.ndarray:
    if config.input_layer == "X":
        return _to_numpy(adata.X)
    return _to_numpy(adata.layers[config.input_layer])


def _normalize_obs_values(adata: ad.AnnData, key: str) -> list[str]:
    return adata.obs[key].astype("string").fillna("").astype(str).tolist()


def _parse_dose_like_value(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
        for separator in ("|", ";", ","):
            if separator in stripped:
                return [float(part.strip()) for part in stripped.split(separator) if part.strip()]
        return [float(stripped)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    return [float(value)]


def _parse_duration_like_value(value: Any) -> list[float]:
    """Parse per-component durations while preserving unknown values as NaN."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [float("nan") if item is None else float(item) for item in parsed]
        for separator in ("|", ";", ","):
            if separator in stripped:
                return [
                    float("nan") if part.strip().lower() in {"", "na", "nan", "none", "null"} else float(part.strip())
                    for part in stripped.split(separator)
                ]
        if stripped.lower() in {"na", "nan", "none", "null"}:
            return [float("nan")]
        return [float(stripped)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float("nan") if item is None else float(item) for item in value]
    return [float(value)]


def _serialize_components(components: list[str]) -> str:
    return "+".join(components)


def _ensure_component_columns(
    adata: ad.AnnData,
    *,
    treatment_col: str,
    dose_col: str | None,
    components_col: str,
    component_doses_col: str,
    component_units_col: str,
    component_durations_col: str,
) -> None:
    if components_col not in adata.obs.columns:
        component_rows: list[list[str]] = []
        for raw in adata.obs[treatment_col].tolist():
            components = _parse_component_like_value(raw)
            if not components:
                label = str(raw).strip()
                components = [label] if label else []
            component_rows.append(components)
        adata.obs[components_col] = component_rows

    if component_doses_col not in adata.obs.columns:
        dose_rows: list[list[float]] = []
        if dose_col and dose_col in adata.obs.columns:
            for raw_components, raw_dose in zip(adata.obs[components_col].tolist(), adata.obs[dose_col].tolist(), strict=False):
                components = _parse_component_like_value(raw_components)
                doses = _parse_dose_like_value(raw_dose)
                if not doses:
                    doses = [1.0] * len(components)
                elif len(doses) == 1 and len(components) > 1:
                    doses = doses * len(components)
                dose_rows.append(doses)
        else:
            for raw_components in adata.obs[components_col].tolist():
                components = _parse_component_like_value(raw_components)
                dose_rows.append([1.0] * len(components))
        adata.obs[component_doses_col] = dose_rows

    if component_units_col not in adata.obs.columns:
        adata.obs[component_units_col] = [["a.u."] * len(_parse_component_like_value(raw)) for raw in adata.obs[components_col].tolist()]

    if component_durations_col not in adata.obs.columns:
        adata.obs[component_durations_col] = [
            [float("nan")] * len(_parse_component_like_value(raw))
            for raw in adata.obs[components_col].tolist()
        ]


def _canonicalize_component_columns(
    adata: ad.AnnData,
    config: RegenAIPTConfig,
) -> ad.AnnData:
    """Return an AnnData copy with canonical component columns.

    Legacy datasets with one treatment label per cell are represented as a
    one-component treatment, preserving the old API while allowing the neural
    model to consume one common tensor layout.
    """
    required = {
        config.component_key,
        config.component_dose_key,
        config.component_dose_unit_key,
        config.component_duration_key,
    }
    if required.issubset(adata.obs.columns):
        return adata

    canonical = adata.copy()
    if config.treatment_key not in canonical.obs.columns:
        raise ValueError(
            f"Cannot derive treatment components because adata.obs['{config.treatment_key}'] is missing"
        )
    _ensure_component_columns(
        canonical,
        treatment_col=config.treatment_key,
        dose_col=config.dose_key,
        components_col=config.component_key,
        component_doses_col=config.component_dose_key,
        component_units_col=config.component_dose_unit_key,
        component_durations_col=config.component_duration_key,
    )
    return canonical


def prepare_round_specific_adata(
    adata: ad.AnnData,
    round_number: int,
    treatment_key: str | None = None,
    dose_key: str | None = None,
    config: RegenAIPTConfig | None = None,
) -> ad.AnnData:
    cfg = config or RegenAIPTConfig(treatment_key=treatment_key or "treatment", dose_key=dose_key or "dose")
    if round_number not in {1, 2}:
        raise ValueError("round_number must be 1 or 2")
    obs = adata.obs
    has_round = "round" in obs.columns
    has_time = "time_point" in obs.columns
    if not has_round and not has_time:
        raise ValueError("prepare_round_specific_adata requires 'round' and/or 'time_point' in adata.obs")

    if round_number == 1:
        source_mask = np.zeros(adata.n_obs, dtype=bool)
        target_mask = np.zeros(adata.n_obs, dtype=bool)
        if has_time:
            source_mask |= obs["time_point"].astype(str).eq("intermediate").to_numpy()
            target_mask |= obs["time_point"].astype(str).eq("post_round1").to_numpy()
        if has_round:
            source_mask |= obs["round"].astype(float).eq(0).to_numpy()
            target_mask |= obs["round"].astype(float).eq(1).to_numpy()
        source_treatment_col = cfg.round1_treatment_key
        source_dose_col = cfg.round1_dose_key
        source_components_col = cfg.round1_components_key
        source_component_doses_col = cfg.round1_component_doses_key
        source_component_units_col = cfg.round1_component_dose_units_key
        source_component_durations_col = cfg.round1_component_durations_key
        treatment_role = "round1_treatment"
    else:
        source_mask = np.zeros(adata.n_obs, dtype=bool)
        target_mask = np.zeros(adata.n_obs, dtype=bool)
        if has_time:
            source_mask |= obs["time_point"].astype(str).eq("post_round1").to_numpy()
            target_mask |= obs["time_point"].astype(str).eq("post_round2").to_numpy()
        if has_round:
            source_mask |= obs["round"].astype(float).eq(1).to_numpy()
            target_mask |= obs["round"].astype(float).eq(2).to_numpy()
        source_treatment_col = cfg.round2_treatment_key
        source_dose_col = cfg.round2_dose_key
        source_components_col = cfg.round2_components_key
        source_component_doses_col = cfg.round2_component_doses_key
        source_component_units_col = cfg.round2_component_dose_units_key
        source_component_durations_col = cfg.round2_component_durations_key
        treatment_role = "round2_treatment"
        if cfg.round1_treatment_key not in obs.columns and cfg.round1_components_key not in obs.columns:
            raise ValueError("Round 2 preparation requires round1 treatment history as context covariate")

    subset = adata[source_mask | target_mask].copy()
    if source_treatment_col not in subset.obs.columns and source_components_col not in subset.obs.columns:
        raise ValueError(f"Could not resolve treatment columns for round {round_number}")

    if source_treatment_col not in subset.obs.columns and source_components_col in subset.obs.columns:
        subset.obs[source_treatment_col] = [
            _serialize_components(_parse_component_like_value(raw)) for raw in subset.obs[source_components_col].tolist()
        ]
    if source_components_col not in subset.obs.columns:
        _ensure_component_columns(
            subset,
            treatment_col=source_treatment_col,
            dose_col=source_dose_col,
            components_col=source_components_col,
            component_doses_col=source_component_doses_col,
            component_units_col=source_component_units_col,
            component_durations_col=source_component_durations_col,
        )
    elif source_component_durations_col not in subset.obs.columns:
        subset.obs[source_component_durations_col] = [
            [float("nan")] * len(_parse_component_like_value(raw))
            for raw in subset.obs[source_components_col].tolist()
        ]

    resolved_treatment_key = treatment_key or cfg.treatment_key
    resolved_dose_key = dose_key or cfg.dose_key or "dose"
    subset_source_mask = source_mask[source_mask | target_mask]
    treatment_rows: list[str] = []
    component_rows: list[list[str]] = []
    component_dose_rows: list[list[float]] = []
    component_unit_rows: list[list[str]] = []
    component_duration_rows: list[list[float]] = []
    scalar_dose_rows: list[float] = []
    for is_source, raw_components, raw_doses, raw_units, raw_durations in zip(
        subset_source_mask,
        subset.obs[source_components_col].tolist(),
        subset.obs[source_component_doses_col].tolist(),
        subset.obs[source_component_units_col].tolist(),
        subset.obs[source_component_durations_col].tolist(),
        strict=False,
    ):
        if bool(is_source):
            components = [cfg.control_treatment]
            doses = [0.0]
            units = ["a.u."]
            durations = [0.0]
        else:
            components = _parse_component_like_value(raw_components)
            doses = _parse_dose_like_value(raw_doses)
            units = _parse_component_like_value(raw_units)
            durations = _parse_duration_like_value(raw_durations)
        treatment_rows.append(_serialize_components(components))
        component_rows.append(components)
        component_dose_rows.append(doses or [1.0] * len(components))
        component_unit_rows.append(units or ["a.u."] * len(components))
        component_duration_rows.append(
            durations if len(durations) == len(components) else [float("nan")] * len(components)
        )
        scalar_dose_rows.append(float(doses[0]) if doses else 1.0)

    subset.obs[resolved_treatment_key] = treatment_rows
    subset.obs[cfg.component_key] = component_rows
    subset.obs[cfg.component_dose_key] = component_dose_rows
    subset.obs[cfg.component_dose_unit_key] = component_unit_rows
    subset.obs[cfg.component_duration_key] = component_duration_rows
    subset.obs[resolved_dose_key] = scalar_dose_rows

    subset.obs["treatment_role"] = treatment_role
    if "treatment_sequence" not in subset.obs.columns and {cfg.round1_treatment_key, cfg.round2_treatment_key}.issubset(subset.obs.columns):
        subset.obs["treatment_sequence"] = (
            subset.obs[cfg.round1_treatment_key].astype(str) + "->" + subset.obs[cfg.round2_treatment_key].astype(str)
        )
    return subset


def _collect_covariate_keys(adata: ad.AnnData, config: RegenAIPTConfig) -> list[str]:
    keys: list[str] = []
    base_keys = [
        config.line_key,
        config.batch_key,
        config.round_key,
        config.time_key,
        config.treatment_role_key,
        config.cell_type_key,
        *config.covariate_keys,
    ]
    for key in base_keys:
        if key and key in adata.obs.columns and key not in keys:
            keys.append(key)

    if (
        config.treatment_role_key in adata.obs.columns
        and config.round1_treatment_key in adata.obs.columns
        and set(adata.obs[config.treatment_role_key].astype(str).unique()) == {"round2_treatment"}
        and config.round1_treatment_key not in keys
    ):
        keys.append(config.round1_treatment_key)
    return keys


def _make_category_encoders(adata: ad.AnnData, config: RegenAIPTConfig) -> tuple[CategoryEncoder, CategoryEncoder, dict[str, CategoryEncoder]]:
    treatment_labels = _normalize_obs_values(adata, config.treatment_key)
    component_labels: list[str] = []
    for raw in adata.obs[config.component_key].tolist():
        component_labels.extend(_parse_component_like_value(raw))
    treatment_encoder = CategoryEncoder.fit(treatment_labels)
    component_encoder = CategoryEncoder.fit(component_labels or [config.control_treatment])
    covariate_encoders = {
        key: CategoryEncoder.fit(_normalize_obs_values(adata, key))
        for key in _collect_covariate_keys(adata, config)
    }
    return treatment_encoder, component_encoder, covariate_encoders


class RegenAIPTDataset(Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
        config: RegenAIPTConfig,
        treatment_encoder: CategoryEncoder,
        covariate_encoders: dict[str, CategoryEncoder],
        component_encoder: CategoryEncoder | None = None,
        indices: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        require_torch()
        adata = _canonicalize_component_columns(adata, config)
        report = validate_regenai_pt_adata(adata, config)
        if not report.is_valid:
            raise ValueError("Invalid AnnData for RegenAI-PT: " + "; ".join(report.errors))

        self.config = config
        self.indices = np.asarray(indices if indices is not None else np.arange(adata.n_obs), dtype=np.int64)
        self.expression = _get_expression_matrix(adata, config)[self.indices]
        self.treatment_encoder = treatment_encoder
        if component_encoder is None:
            component_values: list[str] = []
            for raw in adata.obs[config.component_key].tolist():
                component_values.extend(_parse_component_like_value(raw))
            component_encoder = CategoryEncoder.fit(component_values or [config.control_treatment])
        self.component_encoder = component_encoder
        self.covariate_encoders = covariate_encoders
        self.treatment_ids = treatment_encoder.transform(_normalize_obs_values(adata, config.treatment_key))[self.indices]

        all_component_ids = np.full((adata.n_obs, config.max_components), 0, dtype=np.int64)
        all_component_doses = np.zeros((adata.n_obs, config.max_components), dtype=np.float32)
        all_component_durations = np.zeros((adata.n_obs, config.max_components), dtype=np.float32)
        all_component_duration_mask = np.zeros((adata.n_obs, config.max_components), dtype=np.float32)
        all_component_mask = np.zeros((adata.n_obs, config.max_components), dtype=np.float32)
        for row_idx, (raw_components, raw_doses, raw_durations) in enumerate(
            zip(
                adata.obs[config.component_key].tolist(),
                adata.obs[config.component_dose_key].tolist(),
                adata.obs[config.component_duration_key].tolist(),
                strict=False,
            )
        ):
            components = _parse_component_like_value(raw_components)
            doses = _parse_dose_like_value(raw_doses)
            durations = _parse_duration_like_value(raw_durations)
            if not doses:
                doses = [1.0] * len(components)
            elif len(doses) == 1 and len(components) > 1:
                doses = doses * len(components)
            for comp_idx, component in enumerate(components[: config.max_components]):
                all_component_ids[row_idx, comp_idx] = self.component_encoder.category_to_id[str(component)]
                all_component_doses[row_idx, comp_idx] = float(doses[comp_idx]) if comp_idx < len(doses) else 1.0
                duration = float(durations[comp_idx]) if comp_idx < len(durations) else float("nan")
                if np.isfinite(duration):
                    if duration < 0.0:
                        raise ValueError("Treatment component durations must be non-negative")
                    all_component_durations[row_idx, comp_idx] = duration
                    all_component_duration_mask[row_idx, comp_idx] = 1.0
                all_component_mask[row_idx, comp_idx] = 1.0

        self.component_ids = all_component_ids[self.indices]
        self.component_doses = all_component_doses[self.indices]
        self.component_durations = all_component_durations[self.indices]
        self.component_duration_mask = all_component_duration_mask[self.indices]
        self.component_mask = all_component_mask[self.indices]

        if config.dose_key and config.dose_key in adata.obs.columns:
            self.dose_values = adata.obs[config.dose_key].astype(float).to_numpy(dtype=np.float32)[self.indices]
        else:
            self.dose_values = np.maximum(self.component_doses.sum(axis=1), 1.0).astype(np.float32)

        self.covariate_ids: dict[str, np.ndarray] = {}
        for key, encoder in covariate_encoders.items():
            self.covariate_ids[key] = encoder.transform(_normalize_obs_values(adata, key))[self.indices]

        self.round_ids = self.covariate_ids.get(config.round_key or "", np.full(len(self.indices), -1, dtype=np.int64))
        self.time_ids = self.covariate_ids.get(config.time_key or "", np.full(len(self.indices), -1, dtype=np.int64))
        self.batch_ids = self.covariate_ids.get(config.batch_key or "", np.full(len(self.indices), -1, dtype=np.int64))
        self.sample_weights = (
            np.asarray(sample_weights, dtype=np.float32)[self.indices]
            if sample_weights is not None
            else np.ones(len(self.indices), dtype=np.float32)
        )
        self.delta_target_lookup: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        self.delta_context_keys: tuple[str, ...] = ()
        self._zero_delta_target = np.zeros(1, dtype=np.float32)
        self.delta_targets_configured = False

    def configure_delta_targets(
        self,
        lookup: dict[tuple[int, tuple[int, ...]], np.ndarray],
        context_keys: tuple[str, ...],
    ) -> None:
        self.delta_target_lookup = lookup
        self.delta_context_keys = context_keys
        self._zero_delta_target = np.zeros(self.expression.shape[1], dtype=np.float32)
        self.delta_targets_configured = True

    def _delta_target_for_row(self, idx: int) -> tuple[np.ndarray, bool]:
        if not self.delta_targets_configured:
            return self._zero_delta_target, False
        context = tuple(
            int(self.covariate_ids[key][idx]) for key in self.delta_context_keys
        )
        target = self.delta_target_lookup.get((int(self.treatment_ids[idx]), context))
        if target is None:
            return self._zero_delta_target, False
        return target, True

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        require_torch()
        delta_target, has_delta_target = self._delta_target_for_row(idx)
        return {
            "x": torch.as_tensor(self.expression[idx], dtype=torch.float32),
            "treatment_id": torch.as_tensor(self.treatment_ids[idx], dtype=torch.long),
            "dose_value": torch.as_tensor(self.dose_values[idx], dtype=torch.float32),
            "component_ids": torch.as_tensor(self.component_ids[idx], dtype=torch.long),
            "component_doses": torch.as_tensor(self.component_doses[idx], dtype=torch.float32),
            "component_durations": torch.as_tensor(self.component_durations[idx], dtype=torch.float32),
            "component_duration_mask": torch.as_tensor(self.component_duration_mask[idx], dtype=torch.float32),
            "component_mask": torch.as_tensor(self.component_mask[idx], dtype=torch.float32),
            "covariate_ids": {
                key: torch.as_tensor(values[idx], dtype=torch.long) for key, values in self.covariate_ids.items()
            },
            "round_id": torch.as_tensor(self.round_ids[idx], dtype=torch.long),
            "time_id": torch.as_tensor(self.time_ids[idx], dtype=torch.long),
            "batch_id": torch.as_tensor(self.batch_ids[idx], dtype=torch.long),
            "sample_weight": torch.as_tensor(self.sample_weights[idx], dtype=torch.float32),
            "delta_target": torch.as_tensor(delta_target, dtype=torch.float32),
            "delta_mask": torch.as_tensor(has_delta_target, dtype=torch.bool),
        }


def build_regenai_pt_dataloaders(
    adata: ad.AnnData,
    config: RegenAIPTConfig,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> RegenAIPTDataBundle:
    require_torch()
    adata = _canonicalize_component_columns(adata, config)
    report = validate_regenai_pt_adata(adata, config)
    if not report.is_valid:
        raise ValueError("Invalid AnnData for RegenAI-PT: " + "; ".join(report.errors))

    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must sum to 1.0")

    treatment_encoder, component_encoder, covariate_encoders = _make_category_encoders(adata, config)
    rng = np.random.default_rng(config.random_seed)
    indices = rng.permutation(adata.n_obs)

    n_train = int(round(adata.n_obs * train_fraction))
    n_val = int(round(adata.n_obs * val_fraction))
    n_train = min(n_train, adata.n_obs)
    n_val = min(n_val, adata.n_obs - n_train)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_dataset = RegenAIPTDataset(adata, config, treatment_encoder, covariate_encoders, component_encoder, indices=train_idx)
    val_dataset = RegenAIPTDataset(adata, config, treatment_encoder, covariate_encoders, component_encoder, indices=val_idx)
    test_dataset = RegenAIPTDataset(adata, config, treatment_encoder, covariate_encoders, component_encoder, indices=test_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    mappings = RegenAIPTMappings(
        treatment_to_id=dict(treatment_encoder.category_to_id),
        id_to_treatment={idx: treatment for treatment, idx in treatment_encoder.category_to_id.items()},
        component_to_id=dict(component_encoder.category_to_id),
        id_to_component={idx: component for component, idx in component_encoder.category_to_id.items()},
        covariate_to_id={key: dict(encoder.category_to_id) for key, encoder in covariate_encoders.items()},
        id_to_covariate={
            key: {idx: category for category, idx in encoder.category_to_id.items()}
            for key, encoder in covariate_encoders.items()
        },
        gene_names=adata.var_names.astype(str).tolist(),
        input_dim=int(adata.n_vars),
        max_components=int(config.max_components),
    )

    return RegenAIPTDataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mappings=mappings,
    )


__all__ = [
    "CategoryEncoder",
    "RegenAIPTConfig",
    "RegenAIPTDataBundle",
    "RegenAIPTDataset",
    "RegenAIPTMappings",
    "build_regenai_pt_dataloaders",
    "prepare_round_specific_adata",
    "require_torch",
]

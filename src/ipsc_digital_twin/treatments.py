from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml

REQUIRED_TREATMENT_FIELDS: tuple[str, ...] = (
    "treatment_id",
    "treatment_name",
    "treatment_type",
    "pathway",
    "molecular_target",
    "dose",
    "dose_unit",
    "duration_hours",
    "vendor",
    "catalog_number",
    "grade",
    "mechanism_notes",
    "safety_notes",
    "cost_estimate",
    "active",
)

ALLOWED_TREATMENT_TYPES: set[str] = {
    "small_molecule",
    "protein",
    "cytokine",
    "antibody",
    "gene_editing",
    "other",
}

ALLOWED_GRADES: set[str] = {"RUO", "GMP", "unknown"}
_COMPONENT_OBS_COLS: tuple[str, ...] = (
    "round1_components",
    "round2_components",
    "round1_treatment",
    "round2_treatment",
)


def _parse_components(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith('[') and stripped.endswith(']'):
            import json

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        for sep in ('|', ';', '+', ','):
            if sep in stripped:
                return [part.strip() for part in stripped.split(sep) if part.strip()]
        return [stripped]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def load_treatment_library(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Treatment library not found: {p}")

    raw = yaml.safe_load(p.read_text())
    if isinstance(raw, dict):
        if "treatments" not in raw:
            raise ValueError("YAML dict must contain top-level 'treatments' list")
        records = raw["treatments"]
    else:
        records = raw

    if not isinstance(records, list):
        raise ValueError("Treatment library must be a list of treatment records")

    normalized = []
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("Each treatment record must be a dictionary")
        cloned = dict(rec)
        if 'components' in cloned:
            cloned['components'] = _parse_components(cloned['components'])
        normalized.append(cloned)

    validate_treatment_library(normalized)
    return normalized


def validate_treatment_library(library: list[dict[str, Any]], strict: bool = True) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, rec in enumerate(library):
        missing = [f for f in REQUIRED_TREATMENT_FIELDS if f not in rec]
        if missing:
            errors.append(f"record[{i}] missing required fields: {', '.join(missing)}")
            continue

        tid = str(rec["treatment_id"])
        if tid in seen_ids:
            errors.append(f"duplicate treatment_id: {tid}")
        seen_ids.add(tid)

        ttype = str(rec["treatment_type"])
        if ttype not in ALLOWED_TREATMENT_TYPES:
            errors.append(f"record[{i}] invalid treatment_type: {ttype}")

        grade = str(rec["grade"])
        if grade not in ALLOWED_GRADES:
            errors.append(f"record[{i}] invalid grade: {grade}")

        if not isinstance(rec["active"], bool):
            errors.append(f"record[{i}] active must be boolean")

        if 'components' in rec and not isinstance(rec['components'], list):
            errors.append(f"record[{i}] components must be a list when provided")

    if strict and errors:
        raise ValueError("Invalid treatment library: " + "; ".join(errors))
    return errors


def map_obs_treatments_to_library(
    adata: ad.AnnData,
    library: list[dict[str, Any]],
    obs_cols: tuple[str, ...] = _COMPONENT_OBS_COLS,
) -> dict[str, Any]:
    obs = getattr(adata, "obs", None)
    if obs is None:
        raise ValueError("adata.obs is required")

    validate_treatment_library(library)

    name_to_id = {str(r["treatment_name"]): str(r["treatment_id"]) for r in library}
    unknown: set[str] = set()
    mapped_columns: list[str] = []

    for col in obs_cols:
        if col not in obs.columns:
            continue
        mapped_col = f"{col}_id"
        mapped_vals: list[str | list[str]] = []
        for raw in obs[col].tolist():
            components = _parse_components(raw)
            if not components:
                mapped_vals.append("UNKNOWN")
                continue
            component_ids = []
            for component in components:
                tid = name_to_id.get(component)
                if tid is None:
                    unknown.add(component)
                    component_ids.append("UNKNOWN")
                else:
                    component_ids.append(tid)
            mapped_vals.append(component_ids if len(component_ids) > 1 else component_ids[0])
        adata.obs[mapped_col] = mapped_vals
        mapped_columns.append(mapped_col)

    if unknown:
        warnings.warn(
            f"Treatments in adata not found in library: {', '.join(sorted(unknown))}",
            UserWarning,
            stacklevel=2,
        )

    return {
        "unknown_treatments": sorted(unknown),
        "mapped_columns": mapped_columns,
    }


def generate_treatment_mechanism_summary(library: list[dict[str, Any]]) -> pd.DataFrame:
    validate_treatment_library(library)
    df = pd.DataFrame(library)
    if 'components' in df.columns:
        df['components'] = df['components'].apply(lambda value: ', '.join(value) if isinstance(value, list) else value)
    cols = [
        "treatment_id",
        "treatment_name",
        "treatment_type",
        "pathway",
        "molecular_target",
        "mechanism_notes",
        "active",
    ]
    if 'components' in df.columns:
        cols.append('components')
    return df[cols].sort_values(["treatment_type", "pathway", "treatment_name"]).reset_index(drop=True)

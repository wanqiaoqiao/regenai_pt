from __future__ import annotations

from itertools import combinations
from typing import Any

import anndata as ad
import pandas as pd

from .forward_model import CurrentStateProfile, TransitionPrediction
from .models.forward_interface import ForwardTransitionModelInterface


def _normalize_components(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split('+') if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def generate_candidate_sequences(
    treatment_library: list[dict[str, Any]] | pd.DataFrame,
    include_single_round: bool = True,
    fixed_round1_treatments: list[str] | None = None,
    fixed_round2_treatments: list[str] | None = None,
    include_pairwise_combinations: bool = True,
) -> list[dict[str, Any]]:
    lib = pd.DataFrame(treatment_library) if not isinstance(treatment_library, pd.DataFrame) else treatment_library.copy()
    if 'active' in lib.columns:
        lib = lib[lib['active'].astype(bool)].copy()
    round1_base = fixed_round1_treatments or lib['treatment_name'].astype(str).tolist()
    round2_base = fixed_round2_treatments or lib['treatment_name'].astype(str).tolist()

    def _expand(values: list[str]) -> list[list[str]]:
        combos = [[value] for value in values]
        if include_pairwise_combinations:
            perturbations = [value for value in values if value.lower() not in {'control', 'vehicle', 'untreated'}]
            combos.extend([list(pair) for pair in combinations(perturbations, 2)])
        return combos

    round1 = _expand(round1_base)
    round2 = _expand(round2_base)
    candidates: list[dict[str, Any]] = []
    if include_single_round:
        for treatment in round1:
            candidates.append({'recommendation_type': 'single_round', 'round1_treatment': treatment, 'round2_treatment': None})
    for r1 in round1:
        for r2 in round2:
            candidates.append({'recommendation_type': 'sequential', 'round1_treatment': r1, 'round2_treatment': r2})
    return candidates


def simulate_candidate_sequence(
    forward_model: ForwardTransitionModelInterface,
    current_input: CurrentStateProfile | dict[str, Any] | ad.AnnData,
    current_profile: CurrentStateProfile,
    round1_treatment: str | list[str],
    round2_treatment: str | list[str] | None = None,
) -> TransitionPrediction:
    if round2_treatment is not None and hasattr(forward_model, 'simulate_sequence'):
        result = forward_model.simulate_sequence(
            current_input,
            {'round1_treatment': round1_treatment, 'round2_treatment': round2_treatment},
            covariates=current_profile.metadata,
        )
        if isinstance(result, dict) and 'predicted_state' in result:
            return result['predicted_state']
    if round2_treatment is None and hasattr(forward_model, 'predict'):
        try:
            return forward_model.predict(current_input, {'round_number': 1, 'round1_treatment': round1_treatment, 'covariates': current_profile.metadata})
        except Exception:
            pass
    return forward_model.predict_future_state(
        current_profile,
        round1_treatment='+'.join(_normalize_components(round1_treatment)),
        round2_treatment=None if round2_treatment is None else '+'.join(_normalize_components(round2_treatment)),
        iPSC_line=str(current_profile.metadata.get('iPSC_line')) if current_profile.metadata.get('iPSC_line') is not None else None,
    )


__all__ = ['generate_candidate_sequences', 'simulate_candidate_sequence']

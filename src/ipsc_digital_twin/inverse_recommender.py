from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata as ad
import pandas as pd

from .inverse_scoring import (
    compute_recommendation_metrics,
    current_state_from_input,
    target_state_from_input,
)
from .models.forward_interface import (
    CurrentStateProfile,
    ForwardTransitionModelInterface,
    TargetStateProfile,
)
from .simulation import generate_candidate_sequences, simulate_candidate_sequence


@dataclass
class RecommendationConstraint:
    require_active: bool = True
    max_sequences: int | None = None
    include_single_round: bool = True
    fixed_round1_treatments: list[str] = field(default_factory=list)
    fixed_round2_treatments: list[str] = field(default_factory=list)


def _normalize_library(treatment_library: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    lib = pd.DataFrame(treatment_library) if not isinstance(treatment_library, pd.DataFrame) else treatment_library.copy()
    if 'treatment_name' not in lib.columns:
        raise ValueError('treatment_library must include treatment_name')
    if 'active' not in lib.columns:
        lib['active'] = True
    if 'pathway' not in lib.columns:
        lib['pathway'] = 'unknown'
    return lib


def derive_condition_score_from_state(predicted_state: dict[str, float]) -> float:
    score = float(predicted_state.get('target_marker_score', 0.0))
    score += float(predicted_state.get('fate_probability', 0.0))
    score += float(predicted_state.get('pseudotime', 0.0))
    score -= float(predicted_state.get('stress_score', 0.0))
    score -= float(predicted_state.get('off_target_score', 0.0))
    return score


def recommend_inverse_treatments(
    current_state: CurrentStateProfile | dict[str, float] | ad.AnnData,
    target_state: TargetStateProfile | dict[str, float] | ad.AnnData,
    treatment_library: list[dict[str, Any]] | pd.DataFrame,
    forward_model: ForwardTransitionModelInterface,
    constraints: RecommendationConstraint | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    lib = _normalize_library(treatment_library)
    options = constraints or RecommendationConstraint()

    current_profile = current_state_from_input(current_state)
    target_profile = target_state_from_input(target_state)
    current_input: CurrentStateProfile | dict[str, Any] | ad.AnnData
    current_input = current_state if isinstance(current_state, ad.AnnData) else current_profile

    if options.require_active:
        lib = lib[lib['active'].astype(bool)].copy()

    candidates = generate_candidate_sequences(
        lib,
        include_single_round=options.include_single_round,
        fixed_round1_treatments=options.fixed_round1_treatments,
        fixed_round2_treatments=options.fixed_round2_treatments,
    )
    if options.max_sequences is not None:
        candidates = candidates[: options.max_sequences]

    pathway_map = dict(zip(lib['treatment_name'].astype(str), lib['pathway'].astype(str), strict=False))
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        round1_components = list(candidate['round1_treatment'])
        round2_components = None if candidate['round2_treatment'] is None else list(candidate['round2_treatment'])
        prediction = simulate_candidate_sequence(
            forward_model=forward_model,
            current_input=current_input,
            current_profile=current_profile,
            round1_treatment=round1_components,
            round2_treatment=round2_components,
        )
        metrics = compute_recommendation_metrics(current_profile, target_profile, prediction)
        round1_label = '+'.join(round1_components)
        round2_label = '' if round2_components is None else '+'.join(round2_components)
        rows.append(
            {
                'recommendation_type': candidate['recommendation_type'],
                'round1_treatment': round1_label,
                'round2_treatment': round2_label,
                'round1_components': round1_components,
                'round2_components': [] if round2_components is None else round2_components,
                'treatment_sequence': round1_label if round2_components is None else f'{round1_label}->{round2_label}',
                'predicted_stage': prediction.stage,
                'pathway_round1': ';'.join(pathway_map.get(component, 'unknown') for component in round1_components),
                'pathway_round2': '' if round2_components is None else ';'.join(pathway_map.get(component, 'unknown') for component in round2_components),
                'derived_condition_score': derive_condition_score_from_state(prediction.state_features),
                **metrics,
                **{f'predicted_{key}': value for key, value in prediction.state_features.items()},
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        ['recommendation_score', 'distance_improvement', 'cosine_similarity_to_target'],
        ascending=[False, False, False],
        kind='mergesort',
    )
    selected = ranked.head(top_n).copy()
    if options.include_single_round and top_n >= 2:
        available_types = set(ranked['recommendation_type'].astype(str))
        selected_types = set(selected['recommendation_type'].astype(str))
        for required_type in ('single_round', 'sequential'):
            if required_type in available_types and required_type not in selected_types:
                best_required = ranked[ranked['recommendation_type'] == required_type].head(1)
                selected = pd.concat([selected.iloc[:-1], best_required], ignore_index=True)
                selected_types.add(required_type)
        selected = selected.sort_values(
            ['recommendation_score', 'distance_improvement', 'cosine_similarity_to_target'],
            ascending=[False, False, False],
            kind='mergesort',
        )
    return selected.reset_index(drop=True)

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import anndata as ad
import numpy as np

from .forward_model import CurrentStateProfile, TargetStateProfile, TransitionPrediction

DEFAULT_STATE_FEATURES = (
    'target_marker_score',
    'stress_score',
    'off_target_score',
    'pseudotime',
    'fate_probability',
)


def _state_features_from_adata(adata: ad.AnnData, features: Iterable[str] = DEFAULT_STATE_FEATURES) -> dict[str, float]:
    payload: dict[str, float] = {}
    for feature in features:
        if feature in adata.obs.columns:
            payload[feature] = float(adata.obs[feature].astype(float).mean())
    return payload


def current_state_from_input(current: CurrentStateProfile | dict[str, float] | ad.AnnData) -> CurrentStateProfile:
    if isinstance(current, CurrentStateProfile):
        return current
    if isinstance(current, ad.AnnData):
        return CurrentStateProfile(
            state_features=_state_features_from_adata(current),
            metadata={'adata': current},
        )
    return CurrentStateProfile(state_features=dict(current))


def target_state_from_input(target: TargetStateProfile | dict[str, float] | ad.AnnData) -> TargetStateProfile:
    if isinstance(target, TargetStateProfile):
        return target
    if isinstance(target, ad.AnnData):
        return TargetStateProfile(
            state_features=_state_features_from_adata(target),
            metadata={'reference_adata': target},
        )
    return TargetStateProfile(state_features=dict(target))


def _safe_mean_expression(matrix: Any) -> np.ndarray | None:
    if matrix is None:
        return None
    if isinstance(matrix, ad.AnnData):
        values = matrix.X
    else:
        values = matrix
    if hasattr(values, 'toarray'):
        values = values.toarray()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr.mean(axis=0)
    return None


def _state_vector(features: dict[str, float], target_keys: Iterable[str]) -> np.ndarray:
    return np.asarray([float(features.get(key, 0.0)) for key in target_keys], dtype=np.float32)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _profile_expression(profile: CurrentStateProfile | TargetStateProfile) -> np.ndarray | None:
    if 'reference_adata' in profile.metadata:
        return _safe_mean_expression(profile.metadata['reference_adata'])
    if 'adata' in profile.metadata:
        return _safe_mean_expression(profile.metadata['adata'])
    if 'expression' in profile.metadata:
        return _safe_mean_expression(profile.metadata['expression'])
    return None


def compute_recommendation_metrics(
    current_state: CurrentStateProfile,
    target_state: TargetStateProfile,
    prediction: TransitionPrediction,
) -> dict[str, float]:
    feature_keys = tuple(dict.fromkeys((*target_state.state_features.keys(), *current_state.state_features.keys(), *prediction.state_features.keys())))
    current_vec = _state_vector(current_state.state_features, feature_keys)
    target_vec = _state_vector(target_state.state_features, feature_keys)
    predicted_vec = _state_vector(prediction.state_features, feature_keys)

    current_expr = _profile_expression(current_state)
    target_expr = _profile_expression(target_state)
    predicted_expr = _safe_mean_expression(prediction.metadata.get('expression'))

    if current_expr is not None and target_expr is not None and predicted_expr is not None and current_expr.shape == target_expr.shape == predicted_expr.shape:
        current_to_target_distance = _euclidean(current_expr, target_expr)
        predicted_to_target_distance = _euclidean(predicted_expr, target_expr)
        cosine_similarity_to_target = _cosine_similarity(predicted_expr, target_expr)
    else:
        current_to_target_distance = _euclidean(current_vec, target_vec)
        predicted_to_target_distance = _euclidean(predicted_vec, target_vec)
        cosine_similarity_to_target = _cosine_similarity(predicted_vec, target_vec)

    distance_improvement = current_to_target_distance - predicted_to_target_distance
    predicted_target_marker_score = float(prediction.state_features.get('target_marker_score', np.nan))
    predicted_stress_score = float(prediction.state_features.get('stress_score', 0.0))
    predicted_off_target_score = float(prediction.state_features.get('off_target_score', 0.0))
    uncertainty = float(prediction.uncertainty)

    recommendation_score = (
        distance_improvement
        + cosine_similarity_to_target
        + 0.5 * (0.0 if np.isnan(predicted_target_marker_score) else predicted_target_marker_score)
        - predicted_stress_score
        - predicted_off_target_score
        - uncertainty
    )

    return {
        'predicted_to_target_distance': predicted_to_target_distance,
        'current_to_target_distance': current_to_target_distance,
        'distance_improvement': distance_improvement,
        'cosine_similarity_to_target': cosine_similarity_to_target,
        'predicted_target_marker_score': predicted_target_marker_score,
        'predicted_stress_score': predicted_stress_score,
        'predicted_off_target_score': predicted_off_target_score,
        'uncertainty': uncertainty,
        'recommendation_score': recommendation_score,
    }


__all__ = [
    'compute_recommendation_metrics',
    'current_state_from_input',
    'target_state_from_input',
]

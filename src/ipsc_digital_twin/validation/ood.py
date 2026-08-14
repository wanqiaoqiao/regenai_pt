from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PopulationMoments:
    mean: np.ndarray
    variance: np.ndarray
    n_cells: int


def _as_2d(matrix: Any) -> Any:
    if matrix is None or len(matrix.shape) != 2:
        raise ValueError("Expression input must be a two-dimensional matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Expression input must contain cells and genes")
    return matrix


def population_moments(matrix: Any) -> PopulationMoments:
    """Compute per-gene population moments for dense or sparse expression."""
    values = _as_2d(matrix)
    mean = np.asarray(values.mean(axis=0)).ravel().astype(np.float64)
    if hasattr(values, "multiply"):
        mean_square = np.asarray(values.multiply(values).mean(axis=0)).ravel()
    else:
        dense = np.asarray(values, dtype=np.float64)
        mean_square = np.mean(np.square(dense), axis=0)
    variance = np.maximum(mean_square - np.square(mean), 0.0)
    return PopulationMoments(mean=mean, variance=variance, n_cells=int(values.shape[0]))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(left, right)[0, 1], nan=0.0))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return _pearson(left_rank, right_rank)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.allclose(observed, predicted):
        return 1.0
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(1.0 - np.sum(np.square(observed - predicted)) / denominator)


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(observed - predicted))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def evaluate_population_ood_prediction(
    source_expression: Any,
    predicted_expression: Any,
    observed_expression: Any,
    gene_names: list[str] | np.ndarray,
    *,
    de_top_k: int = 100,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare an OOD prediction with an independently observed population.

    Cells are not paired. Metrics compare population-level gene means,
    variances, and treatment effects relative to the source population.
    """
    source = population_moments(source_expression)
    predicted = population_moments(predicted_expression)
    observed = population_moments(observed_expression)
    genes = np.asarray(gene_names, dtype=str)
    n_genes = len(genes)
    if not (
        len(source.mean)
        == len(predicted.mean)
        == len(observed.mean)
        == n_genes
    ):
        raise ValueError("Source, predicted, observed, and gene_names widths must match")
    if de_top_k <= 0:
        raise ValueError("de_top_k must be positive")

    observed_delta = observed.mean - source.mean
    predicted_delta = predicted.mean - source.mean
    k = min(de_top_k, n_genes)
    observed_de_idx = np.argsort(np.abs(observed_delta))[-k:]
    predicted_de_idx = np.argsort(np.abs(predicted_delta))[-k:]
    overlap = len(set(observed_de_idx.tolist()) & set(predicted_de_idx.tolist())) / k
    direction = np.sign(observed_delta[observed_de_idx]) == np.sign(
        predicted_delta[observed_de_idx]
    )

    source_rmse = _rmse(observed.mean, source.mean)
    predicted_rmse = _rmse(observed.mean, predicted.mean)
    metrics: dict[str, Any] = {
        "evaluation_level": "population",
        "n_source_cells": source.n_cells,
        "n_predicted_cells": predicted.n_cells,
        "n_observed_cells": observed.n_cells,
        "n_genes": n_genes,
        "de_top_k": k,
        "mean_r2": _r2(observed.mean, predicted.mean),
        "mean_pearson": _pearson(observed.mean, predicted.mean),
        "mean_spearman": _spearman(observed.mean, predicted.mean),
        "mean_rmse": predicted_rmse,
        "variance_r2": _r2(observed.variance, predicted.variance),
        "variance_pearson": _pearson(observed.variance, predicted.variance),
        "variance_rmse": _rmse(observed.variance, predicted.variance),
        "delta_r2": _r2(observed_delta, predicted_delta),
        "delta_pearson": _pearson(observed_delta, predicted_delta),
        "delta_cosine": _cosine(observed_delta, predicted_delta),
        "de_mean_r2": _r2(
            observed.mean[observed_de_idx], predicted.mean[observed_de_idx]
        ),
        "de_delta_pearson": _pearson(
            observed_delta[observed_de_idx], predicted_delta[observed_de_idx]
        ),
        "de_delta_cosine": _cosine(
            observed_delta[observed_de_idx], predicted_delta[observed_de_idx]
        ),
        "de_direction_accuracy": float(np.mean(direction)),
        "de_top_k_overlap": float(overlap),
        "source_to_observed_mean_rmse": source_rmse,
        "predicted_to_observed_mean_rmse": predicted_rmse,
        "mean_rmse_improvement": source_rmse - predicted_rmse,
        "mean_rmse_fractional_improvement": (
            float((source_rmse - predicted_rmse) / source_rmse)
            if source_rmse > np.finfo(np.float64).eps
            else 0.0
        ),
    }
    gene_table = pd.DataFrame(
        {
            "gene": genes,
            "source_mean": source.mean,
            "predicted_mean": predicted.mean,
            "observed_mean": observed.mean,
            "source_variance": source.variance,
            "predicted_variance": predicted.variance,
            "observed_variance": observed.variance,
            "predicted_delta": predicted_delta,
            "observed_delta": observed_delta,
            "absolute_delta_error": np.abs(predicted_delta - observed_delta),
        }
    )
    gene_table["observed_de_rank"] = (
        gene_table["observed_delta"].abs().rank(method="min", ascending=False).astype(int)
    )
    gene_table["is_observed_top_de"] = False
    gene_table.loc[observed_de_idx, "is_observed_top_de"] = True
    return metrics, gene_table


def evaluate_additive_transfer_prediction(
    baseline_expression: Any,
    reference_without_component_expression: Any,
    reference_with_component_expression: Any,
    observed_expression: Any,
    gene_names: list[str] | np.ndarray,
    *,
    de_top_k: int = 100,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a transferred component vector added to a matched endpoint.

    The raw population-mean prediction is
    baseline + (reference_with_component - reference_without_component).
    A clipped prediction is also reported because negative log-normalized
    expression values are not physically meaningful.
    """
    baseline = population_moments(baseline_expression)
    reference_without = population_moments(reference_without_component_expression)
    reference_with = population_moments(reference_with_component_expression)
    observed = population_moments(observed_expression)
    genes = np.asarray(gene_names, dtype=str)
    widths = {
        len(baseline.mean),
        len(reference_without.mean),
        len(reference_with.mean),
        len(observed.mean),
        len(genes),
    }
    if len(widths) != 1:
        raise ValueError("All expression matrices and gene_names must have matching widths")
    if de_top_k <= 0:
        raise ValueError("de_top_k must be positive")

    transferred_effect = reference_with.mean - reference_without.mean
    observed_effect = observed.mean - baseline.mean
    predicted_raw = baseline.mean + transferred_effect
    predicted_clipped = np.clip(predicted_raw, 0.0, None)
    k = min(de_top_k, len(genes))
    observed_de_idx = np.argsort(np.abs(observed_effect))[-k:]
    predicted_de_idx = np.argsort(np.abs(transferred_effect))[-k:]
    top_k_overlap = len(
        set(observed_de_idx.tolist()) & set(predicted_de_idx.tolist())
    ) / k
    direction_accuracy = np.mean(
        np.sign(observed_effect[observed_de_idx])
        == np.sign(transferred_effect[observed_de_idx])
    )

    baseline_rmse = _rmse(observed.mean, baseline.mean)
    raw_rmse = _rmse(observed.mean, predicted_raw)
    clipped_rmse = _rmse(observed.mean, predicted_clipped)
    metrics: dict[str, Any] = {
        "evaluation_level": "population_mean",
        "prediction_formula": (
            "baseline + (reference_with_component - reference_without_component)"
        ),
        "n_baseline_cells": baseline.n_cells,
        "n_reference_without_component_cells": reference_without.n_cells,
        "n_reference_with_component_cells": reference_with.n_cells,
        "n_observed_cells": observed.n_cells,
        "n_genes": len(genes),
        "de_top_k": k,
        "negative_raw_prediction_genes": int(np.sum(predicted_raw < 0.0)),
        "negative_raw_prediction_fraction": float(np.mean(predicted_raw < 0.0)),
        "baseline_mean_r2": _r2(observed.mean, baseline.mean),
        "baseline_mean_pearson": _pearson(observed.mean, baseline.mean),
        "baseline_to_observed_mean_rmse": baseline_rmse,
        "additive_raw_mean_r2": _r2(observed.mean, predicted_raw),
        "additive_raw_mean_pearson": _pearson(observed.mean, predicted_raw),
        "additive_raw_to_observed_mean_rmse": raw_rmse,
        "additive_raw_rmse_improvement": baseline_rmse - raw_rmse,
        "additive_raw_fractional_rmse_improvement": (
            float((baseline_rmse - raw_rmse) / baseline_rmse)
            if baseline_rmse > np.finfo(np.float64).eps
            else 0.0
        ),
        "additive_clipped_mean_r2": _r2(observed.mean, predicted_clipped),
        "additive_clipped_mean_pearson": _pearson(observed.mean, predicted_clipped),
        "additive_clipped_to_observed_mean_rmse": clipped_rmse,
        "additive_clipped_rmse_improvement": baseline_rmse - clipped_rmse,
        "additive_clipped_fractional_rmse_improvement": (
            float((baseline_rmse - clipped_rmse) / baseline_rmse)
            if baseline_rmse > np.finfo(np.float64).eps
            else 0.0
        ),
        "component_effect_r2": _r2(observed_effect, transferred_effect),
        "component_effect_pearson": _pearson(observed_effect, transferred_effect),
        "component_effect_cosine": _cosine(observed_effect, transferred_effect),
        "top_de_direction_accuracy": float(direction_accuracy),
        "top_de_overlap": float(top_k_overlap),
        "top_de_component_effect_r2": _r2(
            observed_effect[observed_de_idx],
            transferred_effect[observed_de_idx],
        ),
        "top_de_component_effect_pearson": _pearson(
            observed_effect[observed_de_idx],
            transferred_effect[observed_de_idx],
        ),
        "top_de_component_effect_cosine": _cosine(
            observed_effect[observed_de_idx],
            transferred_effect[observed_de_idx],
        ),
        "top_de_baseline_rmse": _rmse(
            observed.mean[observed_de_idx], baseline.mean[observed_de_idx]
        ),
        "top_de_additive_raw_rmse": _rmse(
            observed.mean[observed_de_idx], predicted_raw[observed_de_idx]
        ),
        "top_de_additive_clipped_rmse": _rmse(
            observed.mean[observed_de_idx], predicted_clipped[observed_de_idx]
        ),
        "top_de_additive_raw_mean_r2": _r2(
            observed.mean[observed_de_idx], predicted_raw[observed_de_idx]
        ),
        "top_de_additive_clipped_mean_r2": _r2(
            observed.mean[observed_de_idx], predicted_clipped[observed_de_idx]
        ),
    }
    gene_table = pd.DataFrame(
        {
            "gene": genes,
            "baseline_mean": baseline.mean,
            "reference_without_component_mean": reference_without.mean,
            "reference_with_component_mean": reference_with.mean,
            "transferred_component_effect": transferred_effect,
            "predicted_raw_mean": predicted_raw,
            "predicted_clipped_mean": predicted_clipped,
            "observed_mean": observed.mean,
            "observed_component_effect": observed_effect,
            "raw_prediction_error": predicted_raw - observed.mean,
            "clipped_prediction_error": predicted_clipped - observed.mean,
        }
    )
    gene_table["observed_de_rank"] = (
        gene_table["observed_component_effect"]
        .abs()
        .rank(method="min", ascending=False)
        .astype(int)
    )
    gene_table["is_observed_top_de"] = False
    gene_table.loc[observed_de_idx, "is_observed_top_de"] = True
    gene_table["is_transferred_top_de"] = False
    gene_table.loc[predicted_de_idx, "is_transferred_top_de"] = True
    return metrics, gene_table


__all__ = [
    "PopulationMoments",
    "evaluate_additive_transfer_prediction",
    "evaluate_population_ood_prediction",
    "population_moments",
]

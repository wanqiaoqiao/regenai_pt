from __future__ import annotations

import numpy as np
import pytest

from ipsc_digital_twin.validation.ood import (
    evaluate_additive_transfer_prediction,
    evaluate_population_ood_prediction,
)


def test_population_ood_metrics_are_perfect_for_matching_population_moments() -> None:
    source = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    observed = np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 0.0]])
    predicted = observed[::-1].copy()

    metrics, genes = evaluate_population_ood_prediction(
        source,
        predicted,
        observed,
        ["G1", "G2", "G3"],
        de_top_k=2,
    )

    assert metrics["mean_r2"] == pytest.approx(1.0)
    assert metrics["variance_r2"] == pytest.approx(1.0)
    assert metrics["delta_cosine"] == pytest.approx(1.0)
    assert metrics["de_direction_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_rmse_improvement"] > 0.0
    assert len(genes) == 3
    assert genes["is_observed_top_de"].sum() == 2


def test_population_ood_metrics_allow_different_cell_counts() -> None:
    source = np.zeros((2, 2))
    predicted = np.ones((3, 2))
    observed = np.ones((5, 2))

    metrics, _ = evaluate_population_ood_prediction(
        source,
        predicted,
        observed,
        ["G1", "G2"],
        de_top_k=2,
    )

    assert metrics["n_source_cells"] == 2
    assert metrics["n_predicted_cells"] == 3
    assert metrics["n_observed_cells"] == 5
    assert metrics["mean_rmse"] == pytest.approx(0.0)


def test_population_ood_metrics_reject_mismatched_genes() -> None:
    with pytest.raises(ValueError, match="widths must match"):
        evaluate_population_ood_prediction(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            ["G1"],
        )


def test_additive_transfer_prediction_recovers_known_component_effect() -> None:
    baseline = np.array([[1.0, 2.0], [1.0, 2.0]])
    reference_without = np.array([[2.0, 2.0], [2.0, 2.0]])
    reference_with = np.array([[3.0, 1.5], [3.0, 1.5]])
    observed = np.array([[2.0, 1.5], [2.0, 1.5]])

    metrics, genes = evaluate_additive_transfer_prediction(
        baseline,
        reference_without,
        reference_with,
        observed,
        ["G1", "G2"],
        de_top_k=2,
    )

    assert metrics["additive_raw_mean_r2"] == pytest.approx(1.0)
    assert metrics["component_effect_cosine"] == pytest.approx(1.0)
    assert metrics["top_de_direction_accuracy"] == pytest.approx(1.0)
    assert metrics["additive_raw_to_observed_mean_rmse"] == pytest.approx(0.0)
    assert genes["predicted_raw_mean"].tolist() == pytest.approx([2.0, 1.5])

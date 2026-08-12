from __future__ import annotations

import pytest


def test_normalized_reconstruction_quality_is_bounded() -> None:
    from ipsc_digital_twin.tuning import normalized_reconstruction_quality

    assert normalized_reconstruction_quality(0.04, 0.08) == pytest.approx(0.5)
    assert normalized_reconstruction_quality(0.0, 0.08) == pytest.approx(1.0)
    assert normalized_reconstruction_quality(0.12, 0.08) == pytest.approx(0.0)


def test_validation_score_uses_agreed_weights() -> None:
    from ipsc_digital_twin.tuning import compute_validation_score

    result = compute_validation_score(
        r2_de=0.4,
        r2_mean=0.9,
        perturbation_fidelity=0.5,
        reconstruction_loss=0.04,
        reference_reconstruction_loss=0.08,
        epoch=12,
    )

    expected = 0.4 * 0.4 + 0.3 * 0.9 + 0.2 * 0.5 + 0.1 * 0.5
    assert result.score == pytest.approx(expected)
    assert result.epoch == 12


def test_best_history_score_respects_minimum_epoch() -> None:
    from ipsc_digital_twin.tuning import best_history_validation_score

    history = {
        "val_mean_DE": [0.9, 0.2, 0.6],
        "val_mean": [0.9, 0.7, 0.8],
        "val_perturbation_fidelity": [0.9, 0.1, 0.7],
        "val_reconstruction_loss": [0.02, 0.07, 0.04],
    }

    result = best_history_validation_score(
        history,
        reference_reconstruction_loss=0.08,
        min_epoch=2,
    )

    assert result.epoch == 3


def test_combined_score_weights_rounds_equally() -> None:
    from ipsc_digital_twin.tuning import (
        combined_round_validation_score,
        compute_validation_score,
    )

    round1 = compute_validation_score(
        r2_de=1.0,
        r2_mean=1.0,
        perturbation_fidelity=1.0,
        reconstruction_loss=0.0,
        reference_reconstruction_loss=0.08,
    )
    round2 = compute_validation_score(
        r2_de=0.0,
        r2_mean=0.0,
        perturbation_fidelity=0.0,
        reconstruction_loss=0.08,
        reference_reconstruction_loss=0.08,
    )

    assert combined_round_validation_score(round1, round2) == pytest.approx(0.5)


def test_tuning_config_rejects_too_few_epochs() -> None:
    from ipsc_digital_twin.tuning import OptunaTuningConfig

    with pytest.raises(ValueError, match="at least 20"):
        OptunaTuningConfig(trial_epochs=10)

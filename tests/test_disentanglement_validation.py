from __future__ import annotations

import json

import numpy as np

from ipsc_digital_twin.validation.disentanglement import (
    evaluate_covariate_leakage,
    evaluate_perturbation_prediction,
    evaluate_reconstruction_quality,
    evaluate_treatment_leakage,
    write_regenai_pt_validation_report,
)


def test_leakage_metrics_run_on_mock_data() -> None:
    z_basal = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [1.1, 0.9],
            [0.0, 0.2],
            [1.2, 1.0],
        ],
        dtype=float,
    )
    treatment_labels = np.array(['control', 'control', 'drug_a', 'drug_a', 'control', 'drug_a'])
    covariate_labels = {
        'iPSC_line': np.array(['line1', 'line1', 'line2', 'line2', 'line1', 'line2']),
        'batch': np.array(['b1', 'b1', 'b2', 'b2', 'b1', 'b2']),
    }

    treatment_result = evaluate_treatment_leakage(z_basal, treatment_labels)
    covariate_result = evaluate_covariate_leakage(z_basal, covariate_labels)

    assert 0.0 <= treatment_result['accuracy'] <= 1.0
    assert treatment_result['n_samples'] == z_basal.shape[0]
    assert set(covariate_result) == {'iPSC_line', 'batch'}
    assert all(0.0 <= metrics['accuracy'] <= 1.0 for metrics in covariate_result.values())


def test_reconstruction_and_perturbation_metrics_are_finite() -> None:
    x_true = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=float)
    x_pred = np.array([[0.9, 2.1, 2.9], [0.6, 1.4, 2.4]], dtype=float)
    x_control = np.array([[0.8, 1.8, 2.8], [0.4, 1.4, 2.4]], dtype=float)

    reconstruction = evaluate_reconstruction_quality(x_true, x_pred)
    perturbation = evaluate_perturbation_prediction(x_true, x_pred, x_control=x_control)

    assert np.isfinite(reconstruction['mse'])
    assert np.isfinite(reconstruction['correlation'])
    assert np.isfinite(reconstruction['gene_level_rmse_mean'])
    assert np.isfinite(perturbation['mse'])
    assert np.isfinite(perturbation['correlation'])
    assert np.isfinite(perturbation['delta_mse_vs_control'])
    assert np.isfinite(perturbation['delta_correlation_vs_control'])


def test_validation_report_is_generated(tmp_path) -> None:
    treatment_result = {
        'n_samples': 6,
        'n_classes': 2,
        'accuracy': 0.5,
        'majority_baseline_accuracy': 0.5,
        'class_counts': {'control': 3, 'drug_a': 3},
    }
    covariate_result = {
        'iPSC_line': {
            'n_samples': 6,
            'n_classes': 2,
            'accuracy': 0.5,
            'majority_baseline_accuracy': 0.5,
            'class_counts': {'line1': 3, 'line2': 3},
        },
    }
    reconstruction = {
        'mse': 0.01,
        'correlation': 0.95,
        'gene_level_rmse_mean': 0.1,
        'gene_level_rmse_std': 0.01,
    }
    perturbation = {
        'mse': 0.02,
        'correlation': 0.9,
        'gene_level_rmse_mean': 0.11,
        'gene_level_rmse_std': 0.02,
        'delta_mse_vs_control': 0.03,
        'delta_correlation_vs_control': 0.88,
    }

    outputs = write_regenai_pt_validation_report(
        tmp_path,
        treatment_leakage=treatment_result,
        covariate_leakage=covariate_result,
        reconstruction_quality=reconstruction,
        perturbation_prediction=perturbation,
    )

    for path in outputs.values():
        assert path
        assert tmp_path.joinpath(path.split('/')[-1]).exists()

    payload = json.loads(tmp_path.joinpath('regenai_pt_validation.json').read_text(encoding='utf-8'))
    assert 'treatment_leakage' in payload
    assert 'covariate_leakage' in payload
    assert 'reconstruction_quality' in payload

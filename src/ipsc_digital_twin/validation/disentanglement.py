from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LeakageEvaluationResult:
    n_samples: int
    n_classes: int
    accuracy: float
    majority_baseline_accuracy: float
    class_counts: dict[str, int]


@dataclass
class ReconstructionEvaluationResult:
    mse: float
    correlation: float
    gene_level_rmse_mean: float
    gene_level_rmse_std: float


@dataclass
class PerturbationPredictionResult:
    mse: float
    correlation: float
    gene_level_rmse_mean: float
    gene_level_rmse_std: float
    delta_mse_vs_control: float | None = None
    delta_correlation_vs_control: float | None = None


def _to_numpy_2d(array_like: Any) -> np.ndarray:
    if hasattr(array_like, 'toarray'):
        array_like = array_like.toarray()
    array = np.asarray(array_like, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError('Expected a 2D array-like input.')
    return array


def _to_string_labels(labels: Any) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError('Expected a 1D label array.')
    if len(array) == 0:
        raise ValueError('Labels must not be empty.')
    normalized = pd.Series(array).fillna('MISSING').astype(str).to_numpy()
    if np.any(pd.Series(normalized).str.strip() == ''):
        raise ValueError('Labels must not contain empty strings.')
    return normalized


def _train_test_indices(
    n_samples: int,
    random_state: int = 0,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError('At least two samples are required for leakage evaluation.')
    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_test = min(n_test, n_samples - 1)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def _nearest_centroid_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = 0,
) -> LeakageEvaluationResult:
    train_idx, test_idx = _train_test_indices(len(labels), random_state=random_state)
    train_x = features[train_idx]
    test_x = features[test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]

    unique_labels = sorted(np.unique(labels).tolist())
    centroids: dict[str, np.ndarray] = {}
    for label in unique_labels:
        label_mask = train_y == label
        if not np.any(label_mask):
            label_mask = labels == label
            centroids[label] = features[label_mask].mean(axis=0)
        else:
            centroids[label] = train_x[label_mask].mean(axis=0)

    classes = np.asarray(unique_labels)
    centroid_stack = np.vstack([centroids[label] for label in classes])
    distances = np.linalg.norm(test_x[:, None, :] - centroid_stack[None, :, :], axis=2)
    predicted = classes[np.argmin(distances, axis=1)]
    accuracy = float(np.mean(predicted == test_y))

    label_series = pd.Series(labels)
    class_counts = label_series.value_counts().sort_index()
    majority_baseline_accuracy = float(class_counts.max() / len(labels))
    return LeakageEvaluationResult(
        n_samples=len(labels),
        n_classes=len(classes),
        accuracy=accuracy,
        majority_baseline_accuracy=majority_baseline_accuracy,
        class_counts={str(key): int(value) for key, value in class_counts.items()},
    )


def evaluate_treatment_leakage(
    z_basal: Any,
    treatment_labels: Any,
    random_state: int = 0,
) -> dict[str, Any]:
    features = _to_numpy_2d(z_basal)
    labels = _to_string_labels(treatment_labels)
    if features.shape[0] != len(labels):
        raise ValueError('z_basal and treatment_labels must have the same number of rows.')
    result = _nearest_centroid_accuracy(features, labels, random_state=random_state)
    return asdict(result)


def evaluate_covariate_leakage(
    z_basal: Any,
    covariate_labels: Any,
    random_state: int = 0,
) -> dict[str, Any]:
    features = _to_numpy_2d(z_basal)
    if isinstance(covariate_labels, dict):
        results: dict[str, Any] = {}
        for covariate_name, labels in covariate_labels.items():
            normalized = _to_string_labels(labels)
            if features.shape[0] != len(normalized):
                raise ValueError(
                    f'z_basal and covariate {covariate_name!r} must have the same number of rows.',
                )
            results[covariate_name] = asdict(
                _nearest_centroid_accuracy(features, normalized, random_state=random_state),
            )
        return results

    labels = _to_string_labels(covariate_labels)
    if features.shape[0] != len(labels):
        raise ValueError('z_basal and covariate_labels must have the same number of rows.')
    return {'covariate': asdict(_nearest_centroid_accuracy(features, labels, random_state=random_state))}


def _flatten_correlation(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    x_true_flat = x_true.ravel()
    x_pred_flat = x_pred.ravel()
    if np.std(x_true_flat) == 0.0 or np.std(x_pred_flat) == 0.0:
        return 0.0
    correlation = np.corrcoef(x_true_flat, x_pred_flat)[0, 1]
    return float(np.nan_to_num(correlation, nan=0.0))


def evaluate_reconstruction_quality(x_true: Any, x_pred: Any) -> dict[str, Any]:
    true_array = _to_numpy_2d(x_true)
    pred_array = _to_numpy_2d(x_pred)
    if true_array.shape != pred_array.shape:
        raise ValueError('x_true and x_pred must have the same shape.')

    residual = true_array - pred_array
    gene_rmse = np.sqrt(np.mean(np.square(residual), axis=0))
    result = ReconstructionEvaluationResult(
        mse=float(np.mean(np.square(residual))),
        correlation=_flatten_correlation(true_array, pred_array),
        gene_level_rmse_mean=float(np.mean(gene_rmse)),
        gene_level_rmse_std=float(np.std(gene_rmse)),
    )
    return asdict(result)


def evaluate_perturbation_prediction(
    x_observed_treated: Any,
    x_pred_treated: Any,
    x_control: Any | None = None,
) -> dict[str, Any]:
    base_metrics = evaluate_reconstruction_quality(x_observed_treated, x_pred_treated)
    result = PerturbationPredictionResult(**base_metrics)

    if x_control is not None:
        control_array = _to_numpy_2d(x_control)
        observed_array = _to_numpy_2d(x_observed_treated)
        predicted_array = _to_numpy_2d(x_pred_treated)
        if control_array.shape != observed_array.shape:
            raise ValueError('x_control must have the same shape as observed/predicted treated matrices.')
        observed_delta = observed_array - control_array
        predicted_delta = predicted_array - control_array
        delta_metrics = evaluate_reconstruction_quality(observed_delta, predicted_delta)
        result.delta_mse_vs_control = float(delta_metrics['mse'])
        result.delta_correlation_vs_control = float(delta_metrics['correlation'])

    return asdict(result)


def write_regenai_pt_validation_report(
    output_dir: str | Path,
    treatment_leakage: dict[str, Any],
    covariate_leakage: dict[str, Any],
    reconstruction_quality: dict[str, Any],
    perturbation_prediction: dict[str, Any] | None = None,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    payload = {
        'treatment_leakage': treatment_leakage,
        'covariate_leakage': covariate_leakage,
        'reconstruction_quality': reconstruction_quality,
        'perturbation_prediction': perturbation_prediction,
    }
    json_path = destination / 'regenai_pt_validation.json'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    markdown_lines = [
        '# RegenAI-PT Validation Report',
        '',
        '## Treatment Leakage',
        f"- Accuracy: {treatment_leakage.get('accuracy', 0.0):.4f}",
        f"- Majority baseline accuracy: {treatment_leakage.get('majority_baseline_accuracy', 0.0):.4f}",
        '',
        '## Covariate Leakage',
    ]
    for covariate_name, metrics in covariate_leakage.items():
        markdown_lines.append(
            f"- {covariate_name}: accuracy={metrics.get('accuracy', 0.0):.4f}, majority_baseline={metrics.get('majority_baseline_accuracy', 0.0):.4f}",
        )
    markdown_lines.extend(
        [
            '',
            '## Reconstruction Quality',
            f"- MSE: {reconstruction_quality.get('mse', 0.0):.6f}",
            f"- Correlation: {reconstruction_quality.get('correlation', 0.0):.4f}",
            f"- Gene-level RMSE mean: {reconstruction_quality.get('gene_level_rmse_mean', 0.0):.6f}",
        ],
    )
    if perturbation_prediction is not None:
        markdown_lines.extend(
            [
                '',
                '## Perturbation Prediction',
                f"- MSE: {perturbation_prediction.get('mse', 0.0):.6f}",
                f"- Correlation: {perturbation_prediction.get('correlation', 0.0):.4f}",
            ],
        )
        if perturbation_prediction.get('delta_mse_vs_control') is not None:
            markdown_lines.append(
                f"- Delta MSE vs control: {perturbation_prediction['delta_mse_vs_control']:.6f}",
            )
        if perturbation_prediction.get('delta_correlation_vs_control') is not None:
            markdown_lines.append(
                f"- Delta correlation vs control: {perturbation_prediction['delta_correlation_vs_control']:.4f}",
            )

    markdown_path = destination / 'regenai_pt_validation.md'
    markdown_path.write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')

    csv_rows = [
        {'metric': 'treatment_accuracy', 'value': treatment_leakage.get('accuracy', 0.0)},
        {'metric': 'reconstruction_mse', 'value': reconstruction_quality.get('mse', 0.0)},
        {'metric': 'reconstruction_correlation', 'value': reconstruction_quality.get('correlation', 0.0)},
    ]
    for covariate_name, metrics in covariate_leakage.items():
        csv_rows.append({'metric': f'{covariate_name}_accuracy', 'value': metrics.get('accuracy', 0.0)})
    if perturbation_prediction is not None:
        csv_rows.append({'metric': 'perturbation_mse', 'value': perturbation_prediction.get('mse', 0.0)})
        csv_rows.append({'metric': 'perturbation_correlation', 'value': perturbation_prediction.get('correlation', 0.0)})
    csv_path = destination / 'regenai_pt_validation_metrics.csv'
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return {
        'json': str(json_path),
        'markdown': str(markdown_path),
        'csv': str(csv_path),
    }

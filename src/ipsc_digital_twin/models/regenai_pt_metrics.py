from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EXTENDED_TRAINING_METRICS = (
    "mean",
    "mean_DE",
    "Var",
    "Var_DE",
    "perturbation_disent",
    "cell_type_disent",
)


def select_treatment_de_genes(
    expression: np.ndarray,
    treatment_ids: np.ndarray,
    control_id: int,
    n_top_genes: int,
) -> dict[int, np.ndarray]:
    """Select top absolute mean-shift genes for each treatment versus control."""
    matrix = np.asarray(expression, dtype=np.float32)
    labels = np.asarray(treatment_ids, dtype=np.int64)
    if matrix.ndim != 2 or labels.ndim != 1 or matrix.shape[0] != labels.shape[0]:
        raise ValueError("expression must be 2D and align with one-dimensional treatment_ids")
    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be positive")

    control_mask = labels == int(control_id)
    if not np.any(control_mask):
        return {}

    def masked_mean(mask: np.ndarray) -> np.ndarray:
        row_indices = np.flatnonzero(mask)
        total = np.zeros(matrix.shape[1], dtype=np.float64)
        for start in range(0, len(row_indices), 1024):
            total += matrix[row_indices[start : start + 1024]].sum(axis=0, dtype=np.float64)
        return total / len(row_indices)

    control_mean = masked_mean(control_mask)
    n_select = min(int(n_top_genes), matrix.shape[1])
    result: dict[int, np.ndarray] = {}
    for treatment_id in np.unique(labels):
        treatment = int(treatment_id)
        if treatment == int(control_id):
            continue
        mask = labels == treatment
        if not np.any(mask):
            continue
        effect = np.abs(masked_mean(mask) - control_mean)
        top_indices = np.argsort(effect)[-n_select:]
        result[treatment] = np.sort(top_indices.astype(np.int64))
    return result


@dataclass
class _GroupMoments:
    n: int
    true_sum: np.ndarray
    predicted_sum: np.ndarray
    true_sum_squares: np.ndarray
    predicted_sum_squares: np.ndarray


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.sum(np.square(observed - observed.mean())))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(observed, predicted) else 0.0
    residual = float(np.sum(np.square(observed - predicted)))
    return float(1.0 - residual / denominator)


def _chance_adjusted_disentanglement(
    correct: int,
    total: int,
    label_counts: dict[int, int],
) -> float:
    if total <= 0 or len(label_counts) < 2:
        return float("nan")
    accuracy = correct / total
    majority_accuracy = max(label_counts.values()) / total
    if majority_accuracy >= 1.0:
        return float("nan")
    excess_accuracy = max(accuracy - majority_accuracy, 0.0)
    return float(np.clip(1.0 - excess_accuracy / (1.0 - majority_accuracy), 0.0, 1.0))


class RegenAIPTEpochMetricAccumulator:
    """Stream distribution and adversarial-probe metrics across one epoch."""

    def __init__(
        self,
        *,
        control_treatment_id: int,
        de_gene_indices: dict[int, np.ndarray],
        cell_type_key: str,
    ) -> None:
        self.control_treatment_id = int(control_treatment_id)
        self.de_gene_indices = de_gene_indices
        self.cell_type_key = cell_type_key
        self.group_moments: dict[int, _GroupMoments] = {}
        self.treatment_correct = 0
        self.treatment_total = 0
        self.treatment_counts: dict[int, int] = {}
        self.cell_type_correct = 0
        self.cell_type_total = 0
        self.cell_type_counts: dict[int, int] = {}

    @staticmethod
    def _update_counts(targets: np.ndarray, counts: dict[int, int]) -> None:
        values, frequencies = np.unique(targets, return_counts=True)
        for value, frequency in zip(values, frequencies, strict=False):
            label = int(value)
            counts[label] = counts.get(label, 0) + int(frequency)

    def update(self, batch: dict[str, Any], outputs: dict[str, Any]) -> None:
        x_true = batch["x"].detach()
        x_predicted = outputs["x_hat"].detach()
        treatment_ids = batch["treatment_id"].detach()

        for treatment_id in treatment_ids.unique():
            treatment = int(treatment_id.item())
            mask = treatment_ids == treatment_id
            n_rows = int(mask.sum().item())
            true_group = x_true[mask]
            predicted_group = x_predicted[mask]
            summaries = (
                true_group.sum(dim=0).double().cpu().numpy(),
                predicted_group.sum(dim=0).double().cpu().numpy(),
                true_group.square().sum(dim=0).double().cpu().numpy(),
                predicted_group.square().sum(dim=0).double().cpu().numpy(),
            )
            existing = self.group_moments.get(treatment)
            if existing is None:
                self.group_moments[treatment] = _GroupMoments(n_rows, *summaries)
            else:
                existing.n += n_rows
                existing.true_sum += summaries[0]
                existing.predicted_sum += summaries[1]
                existing.true_sum_squares += summaries[2]
                existing.predicted_sum_squares += summaries[3]

        treatment_targets = treatment_ids.cpu().numpy()
        treatment_predictions = outputs["treatment_logits_adv"].detach().argmax(dim=1).cpu().numpy()
        self.treatment_correct += int(np.sum(treatment_predictions == treatment_targets))
        self.treatment_total += int(len(treatment_targets))
        self._update_counts(treatment_targets, self.treatment_counts)

        covariate_targets = batch.get("covariate_ids", {}).get(self.cell_type_key)
        covariate_logits = outputs.get("covariate_logits_adv", {}).get(self.cell_type_key)
        if covariate_targets is not None and covariate_logits is not None:
            cell_targets = covariate_targets.detach().cpu().numpy()
            cell_predictions = covariate_logits.detach().argmax(dim=1).cpu().numpy()
            self.cell_type_correct += int(np.sum(cell_predictions == cell_targets))
            self.cell_type_total += int(len(cell_targets))
            self._update_counts(cell_targets, self.cell_type_counts)

    def compute(self) -> dict[str, float]:
        mean_scores: list[float] = []
        mean_de_scores: list[float] = []
        variance_scores: list[float] = []
        variance_de_scores: list[float] = []

        for treatment_id, moments in self.group_moments.items():
            true_mean = moments.true_sum / moments.n
            predicted_mean = moments.predicted_sum / moments.n
            true_variance = np.maximum(
                moments.true_sum_squares / moments.n - np.square(true_mean), 0.0
            )
            predicted_variance = np.maximum(
                moments.predicted_sum_squares / moments.n - np.square(predicted_mean),
                0.0,
            )
            mean_scores.append(_r2_score(true_mean, predicted_mean))
            variance_scores.append(_r2_score(true_variance, predicted_variance))

            de_indices = self.de_gene_indices.get(treatment_id)
            if (
                treatment_id != self.control_treatment_id
                and de_indices is not None
                and len(de_indices) > 0
            ):
                mean_de_scores.append(_r2_score(true_mean[de_indices], predicted_mean[de_indices]))
                variance_de_scores.append(
                    _r2_score(true_variance[de_indices], predicted_variance[de_indices])
                )

        def average(values: list[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        return {
            "mean": average(mean_scores),
            "mean_DE": average(mean_de_scores),
            "Var": average(variance_scores),
            "Var_DE": average(variance_de_scores),
            "perturbation_disent": _chance_adjusted_disentanglement(
                self.treatment_correct,
                self.treatment_total,
                self.treatment_counts,
            ),
            "cell_type_disent": _chance_adjusted_disentanglement(
                self.cell_type_correct,
                self.cell_type_total,
                self.cell_type_counts,
            ),
        }


__all__ = [
    "EXTENDED_TRAINING_METRICS",
    "RegenAIPTEpochMetricAccumulator",
    "select_treatment_de_genes",
]

from __future__ import annotations

import numpy as np
import pytest


def test_de_gene_selection_returns_treatment_specific_indices() -> None:
    from ipsc_digital_twin.models.regenai_pt_metrics import select_treatment_de_genes

    expression = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [5.0, 1.0, 4.0, 1.0],
            [5.0, 1.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    treatment_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)

    selected = select_treatment_de_genes(expression, treatment_ids, control_id=0, n_top_genes=2)

    assert set(selected) == {1}
    assert set(selected[1].tolist()) == {0, 2}


def test_epoch_metrics_are_perfect_for_exact_predictions_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_metrics import (
        RegenAIPTEpochMetricAccumulator,
        select_treatment_de_genes,
    )

    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 5.0, 8.0],
            [3.0, 3.0, 7.0, 9.0],
            [5.0, 2.0, 6.0, 4.0],
            [7.0, 4.0, 8.0, 8.0],
            [9.0, 3.0, 10.0, 9.0],
        ],
        dtype=torch.float32,
    )
    treatment_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    cell_type_ids = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    de_indices = select_treatment_de_genes(
        x.numpy(),
        treatment_ids.numpy(),
        control_id=0,
        n_top_genes=3,
    )
    accumulator = RegenAIPTEpochMetricAccumulator(
        control_treatment_id=0,
        de_gene_indices=de_indices,
        cell_type_key="time_point",
    )
    accumulator.update(
        {
            "x": x,
            "treatment_id": treatment_ids,
            "covariate_ids": {"time_point": cell_type_ids},
        },
        {
            "x_hat": x.clone(),
            "treatment_logits_adv": torch.tensor([[1.0, 0.0]] * len(x)),
            "covariate_logits_adv": {"time_point": torch.tensor([[1.0, 0.0]] * len(x))},
        },
    )

    metrics = accumulator.compute()

    for key in ("mean", "mean_DE", "Var", "Var_DE"):
        assert metrics[key] == pytest.approx(1.0)
    assert metrics["perturbation_disent"] == pytest.approx(1.0)
    assert metrics["cell_type_disent"] == pytest.approx(1.0)
    assert metrics["covariate_adv_accuracy"] == pytest.approx(0.5)
    assert metrics["covariate_adv_accuracy_time_point"] == pytest.approx(0.5)


def test_distribution_metrics_decrease_for_inaccurate_predictions_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_metrics import RegenAIPTEpochMetricAccumulator

    x = torch.tensor(
        [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [3.0, 5.0, 9.0], [4.0, 8.0, 12.0]],
        dtype=torch.float32,
    )
    treatment_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    accumulator = RegenAIPTEpochMetricAccumulator(
        control_treatment_id=0,
        de_gene_indices={1: np.asarray([0, 1, 2])},
        cell_type_key="time_point",
    )
    accumulator.update(
        {
            "x": x,
            "treatment_id": treatment_ids,
            "covariate_ids": {"time_point": torch.tensor([0, 1, 0, 1])},
        },
        {
            "x_hat": torch.zeros_like(x),
            "treatment_logits_adv": torch.tensor([[2.0, 0.0]] * 4),
            "covariate_logits_adv": {"time_point": torch.tensor([[2.0, 0.0]] * 4)},
        },
    )

    metrics = accumulator.compute()

    assert metrics["mean"] < 1.0
    assert metrics["mean_DE"] < 1.0
    assert metrics["Var"] < 1.0
    assert metrics["Var_DE"] < 1.0

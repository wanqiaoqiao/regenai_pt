from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ipsc_digital_twin.validation.metrics import (
    compute_direction_accuracy,
    compute_rmse,
    compute_spearman_ranking_correlation,
    compute_stress_risk_detection_accuracy,
    compute_topk_overlap,
)
from ipsc_digital_twin.validation.reports import generate_validation_report
from ipsc_digital_twin.validation.splits import (
    holdout_by_ipsc_line,
    holdout_by_replicate,
    holdout_by_treatment,
    holdout_by_treatment_sequence,
)


def _toy_split_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round1_treatment": ["A", "A", "B", "B", "C", "C"],
            "iPSC_line": ["L1", "L2", "L1", "L2", "L1", "L2"],
            "treatment_sequence": ["A->X", "A->Y", "B->X", "B->Y", "C->X", "C->Y"],
            "replicate": ["r1", "r2", "r1", "r2", "r1", "r2"],
        }
    )


def test_split_functions_no_leakage() -> None:
    df = _toy_split_df()

    for train, test, held in holdout_by_treatment(df):
        assert held in set(test["round1_treatment"])
        assert held not in set(train["round1_treatment"])

    for train, test, held in holdout_by_ipsc_line(df):
        assert held in set(test["iPSC_line"])
        assert held not in set(train["iPSC_line"])

    for train, test, held in holdout_by_treatment_sequence(df):
        assert held in set(test["treatment_sequence"])
        assert held not in set(train["treatment_sequence"])

    for train, test, held in holdout_by_replicate(df):
        assert held in set(test["replicate"])
        assert held not in set(train["replicate"])


def test_metrics_expected_values_on_toy_data() -> None:
    pred = pd.DataFrame(
        {
            "treatment": ["control", "A", "B", "C"],
            "pred_score": [0.0, 0.8, 0.4, 0.1],
            "pred_stress_risk": [0, 0, 1, 1],
        }
    )
    obs = pd.DataFrame(
        {
            "treatment": ["control", "A", "B", "C"],
            "obs_score": [0.0, 0.7, 0.2, 0.3],
            "stress_risk_label": [0, 0, 1, 0],
        }
    )

    sp = compute_spearman_ranking_correlation(pred, obs, "treatment", "pred_score", "obs_score")
    assert -1.0 <= sp <= 1.0

    rmse = compute_rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    assert abs(rmse - np.sqrt(2.0)) < 1e-9

    topk = compute_topk_overlap(pred, obs, "treatment", "pred_score", "obs_score", k=2)
    assert 0.0 <= topk <= 1.0

    direction = compute_direction_accuracy(pred, obs, "treatment", "pred_score", "obs_score", control_label="control")
    assert 0.0 <= direction <= 1.0

    stress_acc = compute_stress_risk_detection_accuracy(pred, obs, "treatment")
    assert 0.0 <= stress_acc <= 1.0


def test_validation_report_files_created(tmp_path: Path) -> None:
    metrics = {"spearman": 0.5, "rmse": 0.2, "topk_overlap": 0.5}
    pred_obs = pd.DataFrame(
        {
            "treatment": ["A", "B"],
            "pred_score": [0.8, 0.2],
            "obs_score": [0.7, 0.3],
        }
    )

    paths = generate_validation_report(metrics, pred_obs, tmp_path, title="Toy Validation")
    assert Path(paths["json_metrics_path"]).exists()
    assert Path(paths["csv_predictions_path"]).exists()
    assert Path(paths["markdown_report_path"]).exists()

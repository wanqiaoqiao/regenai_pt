from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ipsc_digital_twin.dashboard_export import export_dashboard_artifacts


def test_dashboard_export_creates_all_artifacts_and_json_keys(tmp_path: Path) -> None:
    rankings = pd.DataFrame(
        {
            "treatment_sequence": ["A->B", "C->D"],
            "final_score": [0.9, 0.5],
            "rank": [1, 2],
        }
    )
    clone = pd.DataFrame(
        {
            "clone_id": ["cl1", "cl2"],
            "skew_score": [0.1, 0.4],
            "n_cells": [100, 80],
        }
    )
    doe = pd.DataFrame(
        {
            "condition_id": ["COND_001"],
            "round1_treatment": ["A"],
            "round2_treatment": ["B"],
            "replicate": [1],
            "rationale": ["exploitation_high_final_score"],
        }
    )

    out = export_dashboard_artifacts(
        output_dir=tmp_path,
        experiment_id="exp_001",
        dataset_version="3",
        schema_version="v1.0",
        experiment_summary={"name": "ExpName"},
        qc_summary={"n_cells": 1234},
        treatment_rankings=rankings,
        clone_skew_summary=clone,
        model_metrics={"rmse": 0.2},
        next_doe=doe,
        report_markdown="# Hello\n",
    )

    for f in [
        "experiment_summary.json",
        "qc_summary.json",
        "treatment_rankings.json",
        "treatment_rankings.csv",
        "clone_skew_summary.csv",
        "model_metrics.json",
        "next_doe.csv",
        "report.md",
    ]:
        assert Path(out[f]).exists()

    exp_payload = json.loads(Path(out["experiment_summary.json"]).read_text())
    assert exp_payload["schema_version"] == "v1.0"
    assert exp_payload["experiment_id"] == "exp_001"
    assert exp_payload["dataset_version"] == "3"
    assert "summary" in exp_payload

    qc_payload = json.loads(Path(out["qc_summary.json"]).read_text())
    assert set(["schema_version", "experiment_id", "dataset_version", "qc"]).issubset(set(qc_payload.keys()))

    rankings_payload = json.loads(Path(out["treatment_rankings.json"]).read_text())
    assert "rankings" in rankings_payload
    assert isinstance(rankings_payload["rankings"], list)

    metrics_payload = json.loads(Path(out["model_metrics.json"]).read_text())
    assert "metrics" in metrics_payload

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def export_dashboard_artifacts(
    output_dir: str | Path,
    experiment_id: str,
    dataset_version: str | int,
    schema_version: str,
    experiment_summary: dict[str, Any] | None = None,
    qc_summary: dict[str, Any] | None = None,
    treatment_rankings: pd.DataFrame | None = None,
    clone_skew_summary: pd.DataFrame | None = None,
    model_metrics: dict[str, Any] | None = None,
    next_doe: pd.DataFrame | None = None,
    report_markdown: str | None = None,
) -> dict[str, str]:
    """Export dashboard-ready JSON/CSV/MD artifacts without frontend dependency."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": schema_version,
        "experiment_id": experiment_id,
        "dataset_version": str(dataset_version),
    }

    exp_payload = {**meta, "summary": experiment_summary or {}}
    qc_payload = {**meta, "qc": qc_summary or {}}
    metrics_payload = {**meta, "metrics": model_metrics or {}}

    rankings_df = treatment_rankings.copy() if treatment_rankings is not None else pd.DataFrame(
        columns=["treatment_sequence", "final_score", "rank"]
    )
    rankings_json_payload = {
        **meta,
        "rankings": rankings_df.to_dict(orient="records"),
    }

    clone_df = clone_skew_summary.copy() if clone_skew_summary is not None else pd.DataFrame(
        columns=["clone_id", "skew_score", "n_cells"]
    )
    doe_df = next_doe.copy() if next_doe is not None else pd.DataFrame(
        columns=["condition_id", "round1_treatment", "round2_treatment", "replicate", "rationale"]
    )

    experiment_summary_json = out / "experiment_summary.json"
    qc_summary_json = out / "qc_summary.json"
    treatment_rankings_json = out / "treatment_rankings.json"
    treatment_rankings_csv = out / "treatment_rankings.csv"
    clone_skew_csv = out / "clone_skew_summary.csv"
    model_metrics_json = out / "model_metrics.json"
    next_doe_csv = out / "next_doe.csv"
    report_md = out / "report.md"

    _json_write(experiment_summary_json, exp_payload)
    _json_write(qc_summary_json, qc_payload)
    _json_write(treatment_rankings_json, rankings_json_payload)
    rankings_df.to_csv(treatment_rankings_csv, index=False)
    clone_df.to_csv(clone_skew_csv, index=False)
    _json_write(model_metrics_json, metrics_payload)
    doe_df.to_csv(next_doe_csv, index=False)

    md_text = report_markdown or "# Report\n\nNo narrative report provided."
    report_md.write_text(md_text)

    return {
        "experiment_summary.json": str(experiment_summary_json),
        "qc_summary.json": str(qc_summary_json),
        "treatment_rankings.json": str(treatment_rankings_json),
        "treatment_rankings.csv": str(treatment_rankings_csv),
        "clone_skew_summary.csv": str(clone_skew_csv),
        "model_metrics.json": str(model_metrics_json),
        "next_doe.csv": str(next_doe_csv),
        "report.md": str(report_md),
    }

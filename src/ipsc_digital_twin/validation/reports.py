from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _table_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free markdown table renderer."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join([header, sep] + rows)


def generate_validation_report(
    metrics: dict[str, Any],
    predictions_vs_observed: pd.DataFrame,
    output_dir: str | Path,
    title: str = "Validation Report",
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "validation_metrics.json"
    csv_path = out / "validation_predictions_vs_observed.csv"
    md_path = out / "validation_report.md"

    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    predictions_vs_observed.to_csv(csv_path, index=False)

    md_lines = [f"# {title}", "", "## Metrics"]
    for k, v in metrics.items():
        md_lines.append(f"- {k}: {v}")

    md_lines.extend(["", "## Prediction vs Observed Preview", ""])
    if predictions_vs_observed.empty:
        md_lines.append("No rows available.")
    else:
        preview = predictions_vs_observed.head(10)
        md_lines.append(_table_to_markdown(preview))

    md_path.write_text("\n".join(md_lines) + "\n")

    return {
        "json_metrics_path": str(json_path),
        "csv_predictions_path": str(csv_path),
        "markdown_report_path": str(md_path),
    }

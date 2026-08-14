from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from ipsc_digital_twin.validation.ood import evaluate_additive_transfer_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GSM4876134 + (GSM7147769 - GSM7147768) against "
            "held-out GSM4876138 at the population-mean level."
        )
    )
    parser.add_argument("--training-adata", required=True)
    parser.add_argument("--observed-adata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-sample-id", default="GSM4876134")
    parser.add_argument("--reference-without-sample-id", default="GSM7147768")
    parser.add_argument("--reference-with-sample-id", default="GSM7147769")
    parser.add_argument("--observed-sample-id", default="GSM4876138")
    parser.add_argument("--expression-layer", default="log_normalized")
    parser.add_argument("--de-top-k", type=int, default=100)
    return parser


def _select_sample(adata: ad.AnnData, sample_id: str) -> ad.AnnData:
    if "sample_id" not in adata.obs.columns:
        raise ValueError("AnnData is missing obs['sample_id']")
    selected = adata[adata.obs["sample_id"].astype(str).eq(sample_id)].copy()
    if selected.n_obs == 0:
        raise ValueError(f"No cells found for sample_id={sample_id!r}")
    return selected


def _expression(adata: ad.AnnData, layer: str) -> Any:
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"AnnData is missing expression layer {layer!r}")
    return adata.layers[layer]


def _verify_gene_order(*datasets: ad.AnnData) -> list[str]:
    genes = datasets[0].var_names.astype(str).tolist()
    for dataset in datasets[1:]:
        if dataset.var_names.astype(str).tolist() != genes:
            raise ValueError("All AnnData inputs must have identical gene names and order")
    return genes


def _save_plots(gene_table: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    endpoint_path = output_dir / "additive_predicted_vs_observed_means.png"
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        gene_table["observed_mean"],
        gene_table["predicted_raw_mean"],
        s=5,
        alpha=0.25,
        color="#D55E00",
        label="Raw additive",
    )
    low = float(min(gene_table["observed_mean"].min(), gene_table["predicted_raw_mean"].min()))
    high = float(max(gene_table["observed_mean"].max(), gene_table["predicted_raw_mean"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1)
    ax.set(
        title="Transferred C59 vector: endpoint means",
        xlabel="Observed GSM4876138 mean",
        ylabel="Predicted additive mean",
    )
    fig.tight_layout()
    fig.savefig(endpoint_path, dpi=180)
    plt.close(fig)

    effect_path = output_dir / "transferred_vs_observed_c59_effect.png"
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(
        gene_table["observed_component_effect"],
        gene_table["transferred_component_effect"],
        s=5,
        alpha=0.25,
        color="#0072B2",
    )
    low = float(
        min(
            gene_table["observed_component_effect"].min(),
            gene_table["transferred_component_effect"].min(),
        )
    )
    high = float(
        max(
            gene_table["observed_component_effect"].max(),
            gene_table["transferred_component_effect"].max(),
        )
    )
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1)
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.axvline(0.0, color="#999999", linewidth=0.6)
    ax.set(
        title="C59 effect transfer between experimental contexts",
        xlabel="Observed C59 effect: GSM4876138 - GSM4876134",
        ylabel="Transferred C59 effect: GSM7147769 - GSM7147768",
    )
    fig.tight_layout()
    fig.savefig(effect_path, dpi=180)
    plt.close(fig)
    return {"endpoint_plot": str(endpoint_path), "component_effect_plot": str(effect_path)}


def _write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, str],
) -> Path:
    path = output_dir / "additive_c59_transfer_report.md"
    lines = [
        "# Additive C59 Transfer Evaluation",
        "",
        "## Prediction",
        "",
        "```text",
        "predicted GSM4876138",
        "= mean(GSM4876134)",
        "+ [mean(GSM7147769) - mean(GSM7147768)]",
        "```",
        "",
        "- `GSM4876134` is the matched STAN D28 D-only endpoint baseline.",
        "- `GSM7147769 - GSM7147768` estimates the incremental C59 vector in the CS0007 context.",
        "- `GSM4876138` is the held-out observed STAN D28 D+C59 endpoint.",
        f"- Expression layer: `{metadata['expression_layer']}`",
        "",
        "## Endpoint Results",
        f"- D-only baseline mean R2: {metrics['baseline_mean_r2']:.4f}",
        f"- D-only baseline RMSE: {metrics['baseline_to_observed_mean_rmse']:.6f}",
        f"- Raw additive mean R2: {metrics['additive_raw_mean_r2']:.4f}",
        f"- Raw additive Pearson: {metrics['additive_raw_mean_pearson']:.4f}",
        f"- Raw additive RMSE: {metrics['additive_raw_to_observed_mean_rmse']:.6f}",
        f"- Raw additive fractional RMSE improvement: {metrics['additive_raw_fractional_rmse_improvement']:.2%}",
        f"- Clipped additive mean R2: {metrics['additive_clipped_mean_r2']:.4f}",
        f"- Clipped additive RMSE: {metrics['additive_clipped_to_observed_mean_rmse']:.6f}",
        "",
        "## Incremental C59 Effect",
        f"- All-gene effect Pearson: {metrics['component_effect_pearson']:.4f}",
        f"- All-gene effect cosine: {metrics['component_effect_cosine']:.4f}",
        f"- Top-{metrics['de_top_k']} direction accuracy: {metrics['top_de_direction_accuracy']:.4f}",
        f"- Top-{metrics['de_top_k']} overlap: {metrics['top_de_overlap']:.4f}",
        f"- Top-{metrics['de_top_k']} effect Pearson: {metrics['top_de_component_effect_pearson']:.4f}",
        "",
        "## Expression-Space Warning",
        f"- The exact additive equation produced {metrics['negative_raw_prediction_genes']} negative gene means "
        f"({metrics['negative_raw_prediction_fraction']:.2%}).",
        "- Raw results preserve the exact vector equation. Clipped results replace negative predicted means with zero.",
        "- Addition in log-normalized space is an empirical vector baseline, not a count-generative model.",
        "",
        "## Artifacts",
        *[f"- `{Path(value).name}`" for value in artifacts.values()],
        "",
        "## Limitations",
        "- The C59 vector is transferred across iPSC lines, protocol backgrounds, and samples.",
        "- Sample, condition, and batch are confounded, so this does not isolate a purely causal C59 effect.",
        "- This is a population-mean calculation; individual cells are not paired.",
        "- Results are research outputs and require prospective validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = ad.read_h5ad(args.training_adata)
    observed_all = ad.read_h5ad(args.observed_adata)
    baseline = _select_sample(training, args.baseline_sample_id)
    reference_without = _select_sample(training, args.reference_without_sample_id)
    reference_with = _select_sample(training, args.reference_with_sample_id)
    observed = _select_sample(observed_all, args.observed_sample_id)
    genes = _verify_gene_order(baseline, reference_without, reference_with, observed)

    metrics, gene_table = evaluate_additive_transfer_prediction(
        _expression(baseline, args.expression_layer),
        _expression(reference_without, args.expression_layer),
        _expression(reference_with, args.expression_layer),
        _expression(observed, args.expression_layer),
        genes,
        de_top_k=args.de_top_k,
    )
    gene_path = output_dir / "additive_c59_gene_comparison.csv"
    metrics_path = output_dir / "additive_c59_metrics.json"
    prediction_path = output_dir / "additive_c59_predicted_population_mean.csv"
    gene_table.sort_values("observed_de_rank").to_csv(gene_path, index=False)
    gene_table.loc[
        :, ["gene", "predicted_raw_mean", "predicted_clipped_mean"]
    ].to_csv(prediction_path, index=False)
    metadata = {
        "training_adata": str(Path(args.training_adata).resolve()),
        "observed_adata": str(Path(args.observed_adata).resolve()),
        "baseline_sample_id": args.baseline_sample_id,
        "reference_without_sample_id": args.reference_without_sample_id,
        "reference_with_sample_id": args.reference_with_sample_id,
        "observed_sample_id": args.observed_sample_id,
        "expression_layer": args.expression_layer,
    }
    metrics_path.write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts = {
        "metrics": str(metrics_path),
        "gene_comparison": str(gene_path),
        "predicted_population_mean": str(prediction_path),
        **_save_plots(gene_table, output_dir),
    }
    report_path = _write_report(output_dir, metadata, metrics, artifacts)
    print(json.dumps({"metrics": metrics, "report": str(report_path), **artifacts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

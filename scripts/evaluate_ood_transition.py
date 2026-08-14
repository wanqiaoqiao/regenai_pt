from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
from ipsc_digital_twin.validation.ood import evaluate_population_ood_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained RegenAI-PT round-2 model against an independently "
            "held-out post-round-2 population."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-adata", required=True)
    parser.add_argument("--observed-adata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sample-id", help="Optional source sample_id filter")
    parser.add_argument("--observed-sample-id", help="Optional observed sample_id filter")
    parser.add_argument("--round1-treatment", required=True)
    parser.add_argument("--round2-components", required=True, help="Comma-separated components")
    parser.add_argument("--round2-doses", required=True, help="Comma-separated component doses")
    parser.add_argument("--ipsc-line", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--de-top-k", type=int, default=100)
    parser.add_argument("--pca-cells-per-group", type=int, default=1000)
    parser.add_argument("--pca-top-genes", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _select_sample(adata: ad.AnnData, sample_id: str | None, label: str) -> ad.AnnData:
    if sample_id is None:
        return adata
    if "sample_id" not in adata.obs.columns:
        raise ValueError(f"{label} AnnData is missing obs['sample_id']")
    selected = adata[adata.obs["sample_id"].astype(str).eq(sample_id)].copy()
    if selected.n_obs == 0:
        raise ValueError(f"No {label} cells found for sample_id={sample_id!r}")
    return selected


def _expression(adata: ad.AnnData, layer: str) -> Any:
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"AnnData is missing required expression layer {layer!r}")
    return adata.layers[layer]


def _validate_inputs(
    source: ad.AnnData,
    observed: ad.AnnData,
    adapter: RegenAIPTForwardAdapter,
    components: list[str],
    doses: list[float],
    ipsc_line: str,
    round1_treatment: str,
) -> None:
    if adapter.round2_trainer is None or adapter.round2_trainer.mappings is None:
        raise RuntimeError("The model artifact does not contain a fitted round-2 trainer")
    mappings = adapter.round2_trainer.mappings
    if source.var_names.astype(str).tolist() != mappings.gene_names:
        raise ValueError("Source gene names/order do not match the trained model")
    if observed.var_names.astype(str).tolist() != mappings.gene_names:
        raise ValueError("Observed gene names/order do not match the trained model")
    unknown = sorted(set(components) - mappings.component_to_id.keys())
    if unknown:
        raise ValueError(f"Round-2 components absent from model mappings: {', '.join(unknown)}")
    if len(components) != len(doses):
        raise ValueError("round2-components and round2-doses must have equal lengths")
    if len(components) > mappings.max_components:
        raise ValueError(f"Model supports at most {mappings.max_components} components")
    covariates = mappings.covariate_to_id
    if ipsc_line not in covariates.get("iPSC_line", {}):
        raise ValueError(f"iPSC line is absent from model mappings: {ipsc_line}")
    if round1_treatment not in covariates.get("round1_treatment", {}):
        raise ValueError(
            "Round-1 treatment context is absent from round-2 model mappings: "
            f"{round1_treatment}"
        )


def _predict_in_batches(
    trainer: Any,
    source: ad.AnnData,
    components: list[str],
    doses: list[float],
    covariates: dict[str, Any],
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, source.n_obs, batch_size):
        stop = min(start + batch_size, source.n_obs)
        prediction = trainer.predict_adata(
            source[start:stop],
            treatment=components,
            dose=doses,
            covariates=covariates,
        )
        for key, values in prediction.items():
            collected.setdefault(key, []).append(np.asarray(values))
        print(f"Predicted {stop}/{source.n_obs} source cells", flush=True)
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _to_dense_rows(
    matrix: Any,
    indices: np.ndarray,
    gene_indices: np.ndarray,
) -> np.ndarray:
    selected = matrix[indices]
    selected = selected[:, gene_indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def _pca_coordinates(
    source: Any,
    predicted: np.ndarray,
    observed: Any,
    n_per_group: int,
    n_top_genes: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    def sample_indices(n_rows: int) -> np.ndarray:
        n = min(n_rows, n_per_group)
        return np.sort(rng.choice(n_rows, size=n, replace=False))

    if n_top_genes <= 0:
        raise ValueError("pca-top-genes must be positive")
    source_variance = np.asarray(source.multiply(source).mean(axis=0)).ravel() - np.square(
        np.asarray(source.mean(axis=0)).ravel()
    ) if sparse.issparse(source) else np.var(np.asarray(source), axis=0)
    observed_variance = np.asarray(observed.multiply(observed).mean(axis=0)).ravel() - np.square(
        np.asarray(observed.mean(axis=0)).ravel()
    ) if sparse.issparse(observed) else np.var(np.asarray(observed), axis=0)
    combined_variance = source_variance + np.var(predicted, axis=0) + observed_variance
    gene_indices = np.argsort(combined_variance)[-min(n_top_genes, source.shape[1]):]
    source_values = _to_dense_rows(source, sample_indices(source.shape[0]), gene_indices)
    predicted_values = predicted[sample_indices(predicted.shape[0])][:, gene_indices]
    observed_values = _to_dense_rows(observed, sample_indices(observed.shape[0]), gene_indices)
    combined = np.vstack([source_values, predicted_values, observed_values]).astype(np.float64)
    combined -= combined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(combined, full_matrices=False)
    coordinates = combined @ vt[:2].T
    groups = (
        ["source"] * len(source_values)
        + ["predicted"] * len(predicted_values)
        + ["observed"] * len(observed_values)
    )
    return pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "group": groups})


def _save_plots(gene_table: pd.DataFrame, coordinates: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    colors = {"source": "#5B6770", "predicted": "#D55E00", "observed": "#0072B2"}
    pca_path = output_dir / "ood_population_pca.png"
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for group, table in coordinates.groupby("group", sort=False):
        ax.scatter(table["PC1"], table["PC2"], s=10, alpha=0.35, label=group, color=colors[group])
        center = table[["PC1", "PC2"]].mean()
        ax.scatter(center["PC1"], center["PC2"], s=130, marker="X", color=colors[group], edgecolor="white", linewidth=1.0)
    ax.set(title="OOD population comparison", xlabel="PC1", ylabel="PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pca_path, dpi=180)
    plt.close(fig)

    mean_path = output_dir / "predicted_vs_observed_gene_means.png"
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(gene_table["observed_mean"], gene_table["predicted_mean"], s=5, alpha=0.25, color="#0072B2")
    low = float(min(gene_table["observed_mean"].min(), gene_table["predicted_mean"].min()))
    high = float(max(gene_table["observed_mean"].max(), gene_table["predicted_mean"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1)
    ax.set(title="Predicted vs observed population means", xlabel="Observed gene mean", ylabel="Predicted gene mean")
    fig.tight_layout()
    fig.savefig(mean_path, dpi=180)
    plt.close(fig)

    delta_path = output_dir / "predicted_vs_observed_gene_deltas.png"
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(gene_table["observed_delta"], gene_table["predicted_delta"], s=5, alpha=0.25, color="#D55E00")
    low = float(min(gene_table["observed_delta"].min(), gene_table["predicted_delta"].min()))
    high = float(max(gene_table["observed_delta"].max(), gene_table["predicted_delta"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=1)
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.axvline(0.0, color="#999999", linewidth=0.6)
    ax.set(title="Predicted vs observed treatment effects", xlabel="Observed delta from source", ylabel="Predicted delta from source")
    fig.tight_layout()
    fig.savefig(delta_path, dpi=180)
    plt.close(fig)
    return {"pca_plot": str(pca_path), "mean_plot": str(mean_path), "delta_plot": str(delta_path)}


def _write_report(output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any], paths: dict[str, str]) -> Path:
    report_path = output_dir / "ood_evaluation_report.md"
    report = [
        "# RegenAI-PT OOD Transition Evaluation",
        "",
        "## Experiment",
        f"- Source sample: `{metadata['source_sample_id']}` ({metrics['n_source_cells']} cells)",
        f"- Observed held-out sample: `{metadata['observed_sample_id']}` ({metrics['n_observed_cells']} cells)",
        f"- iPSC line: `{metadata['iPSC_line']}`",
        f"- Round-1 context: `{metadata['round1_treatment']}`",
        f"- Simulated round-2 treatment: `{metadata['round2_treatment']}` at component doses `{metadata['round2_doses']}`",
        "- Evaluation is population-level; cells are not paired because scRNA-seq is destructive.",
        "",
        "## Results",
        f"- Mean-expression R2: {metrics['mean_r2']:.4f}",
        f"- Mean-expression Pearson: {metrics['mean_pearson']:.4f}",
        f"- Variance R2: {metrics['variance_r2']:.4f}",
        f"- Treatment-delta Pearson: {metrics['delta_pearson']:.4f}",
        f"- Treatment-delta cosine: {metrics['delta_cosine']:.4f}",
        f"- Top-{metrics['de_top_k']} DE direction accuracy: {metrics['de_direction_accuracy']:.4f}",
        f"- Top-{metrics['de_top_k']} DE overlap: {metrics['de_top_k_overlap']:.4f}",
        f"- Source-to-observed mean RMSE: {metrics['source_to_observed_mean_rmse']:.6f}",
        f"- Predicted-to-observed mean RMSE: {metrics['predicted_to_observed_mean_rmse']:.6f}",
        f"- Fractional RMSE improvement: {metrics['mean_rmse_fractional_improvement']:.4f}",
        "",
        "## Artifacts",
        *[f"- `{Path(path).name}`" for path in paths.values()],
        "",
        "## Limitations",
        "- This evaluates one held-out condition and does not establish general OOD performance.",
        "- The condition is combination-OOD, but its individual components, line, and round-1 context were represented during training.",
        "- Batch is intentionally not set to the unseen held-out sample ID because the model has no embedding for an unseen batch.",
        "- Predictions are research outputs for prospective experimental validation, not clinical results.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    components = _parse_csv(args.round2_components)
    doses = [float(value) for value in _parse_csv(args.round2_doses)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    adapter = RegenAIPTForwardAdapter.load(args.model_path)
    assert adapter.round2_trainer is not None
    trainer = adapter.round2_trainer
    print(f"Loading source: {args.source_adata}")
    source = _select_sample(ad.read_h5ad(args.source_adata), args.source_sample_id, "source")
    print(f"Loading observed held-out data: {args.observed_adata}")
    observed = _select_sample(ad.read_h5ad(args.observed_adata), args.observed_sample_id, "observed")
    _validate_inputs(source, observed, adapter, components, doses, args.ipsc_line, args.round1_treatment)

    covariates = {
        "iPSC_line": args.ipsc_line,
        "round": "2",
        "time_point": "post_round2",
        "round1_treatment": args.round1_treatment,
        "treatment_role": "round2_treatment",
    }
    # Retain the known source batch. An unseen observed batch cannot be encoded.
    if "batch" in source.obs.columns:
        source_batches = source.obs["batch"].astype(str).unique().tolist()
        if len(source_batches) == 1:
            covariates["batch"] = source_batches[0]

    prediction = _predict_in_batches(
        trainer,
        source,
        components,
        doses,
        covariates,
        args.batch_size,
    )
    input_layer = trainer.config.input_layer
    source_expression = _expression(source, input_layer)
    observed_expression = _expression(observed, input_layer)
    metrics, gene_table = evaluate_population_ood_prediction(
        source_expression,
        prediction["x_hat"],
        observed_expression,
        source.var_names.astype(str).tolist(),
        de_top_k=args.de_top_k,
    )

    expression_path = output_dir / "predicted_expression.npy"
    latent_path = output_dir / "predicted_latent.csv"
    gene_path = output_dir / "gene_level_comparison.csv"
    metrics_path = output_dir / "ood_metrics.json"
    np.save(expression_path, prediction["x_hat"].astype(np.float32))
    pd.DataFrame(
        prediction["z_total"],
        columns=[f"latent_{idx + 1}" for idx in range(prediction["z_total"].shape[1])],
    ).to_csv(latent_path, index=False)
    gene_table.sort_values("observed_de_rank").to_csv(gene_path, index=False)

    coordinates = _pca_coordinates(
        source_expression,
        prediction["x_hat"],
        observed_expression,
        args.pca_cells_per_group,
        args.pca_top_genes,
        args.random_seed,
    )
    pca_csv_path = output_dir / "ood_population_pca.csv"
    coordinates.to_csv(pca_csv_path, index=False)
    plot_paths = _save_plots(gene_table, coordinates, output_dir)
    metadata = {
        "model_path": str(Path(args.model_path).resolve()),
        "source_adata": str(Path(args.source_adata).resolve()),
        "observed_adata": str(Path(args.observed_adata).resolve()),
        "source_sample_id": args.source_sample_id or "all source cells",
        "observed_sample_id": args.observed_sample_id or "all observed cells",
        "iPSC_line": args.ipsc_line,
        "round1_treatment": args.round1_treatment,
        "round2_components": components,
        "round2_doses": doses,
        "round2_treatment": "+".join(components),
        "input_layer": input_layer,
        "covariates": covariates,
    }
    metrics_path.write_text(
        json.dumps({"metadata": metadata, "metrics": metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_paths = {
        "metrics": str(metrics_path),
        "gene_table": str(gene_path),
        "predicted_expression": str(expression_path),
        "predicted_latent": str(latent_path),
        "pca_coordinates": str(pca_csv_path),
        **plot_paths,
    }
    report_path = _write_report(output_dir, metadata, metrics, artifact_paths)
    print(json.dumps({"metrics": metrics, "report": str(report_path), **artifact_paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

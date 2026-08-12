from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad

from ipsc_digital_twin.registry.dataset_registry import DatasetRegistry
from ipsc_digital_twin.tuning import OptunaTuningConfig, run_optuna_tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune combined-round RegenAI-PT hyperparameters with Optuna."
    )
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--trial-epochs", type=int, default=50)
    parser.add_argument("--reference-reconstruction-loss", type=float, default=0.08)
    parser.add_argument("--input-layer", choices=["X", "raw_counts", "log_normalized"], default="log_normalized")
    parser.add_argument("--covariate-keys", default="iPSC_line,batch,round,time_point")
    parser.add_argument("--control-treatment", default="control")
    parser.add_argument("--n-de-genes", type=int, default=100)
    parser.add_argument("--cell-type-key", default="time_point")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--study-name", default="regenai_pt_combined_tuning")
    parser.add_argument(
        "--storage",
        help="Optional Optuna storage URL. Defaults to SQLite in the output directory.",
    )
    return parser


def _load_dataset_record(registry_dir: str | Path, dataset_id: str) -> dict[str, Any]:
    registry = DatasetRegistry(Path(registry_dir) / "datasets.json")
    record = next(
        (item for item in registry.list_datasets() if item["dataset_id"] == dataset_id),
        None,
    )
    if record is None:
        raise KeyError(f"Dataset ID not found in registry: {dataset_id}")
    path = Path(record["anndata_path"])
    if not path.exists():
        raise FileNotFoundError(f"Registered AnnData file does not exist: {path}")
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = _load_dataset_record(args.registry_dir, args.dataset_id)
    covariate_keys = tuple(
        key.strip() for key in args.covariate_keys.split(",") if key.strip()
    )
    tuning = OptunaTuningConfig(
        n_trials=args.n_trials,
        trial_epochs=args.trial_epochs,
        reference_reconstruction_loss=args.reference_reconstruction_loss,
        input_layer=args.input_layer,
        covariate_keys=covariate_keys,
        control_treatment=args.control_treatment,
        n_de_genes=args.n_de_genes,
        cell_type_key=args.cell_type_key,
        device=args.device,
        random_seed=args.random_seed,
        study_name=args.study_name,
        storage=args.storage,
    )

    print(f"Loading dataset {args.dataset_id}: {record['anndata_path']}")
    adata = ad.read_h5ad(record["anndata_path"])
    print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    print(
        f"Starting {tuning.n_trials} Optuna trials; each trial trains round 1 and round 2 "
        f"for up to {tuning.trial_epochs} epochs each."
    )
    result = run_optuna_tuning(adata, args.output_dir, tuning)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

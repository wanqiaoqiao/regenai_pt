from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np


@dataclass(frozen=True, slots=True)
class ValidationScore:
    epoch: int
    r2_de: float
    r2_mean: float
    perturbation_fidelity: float
    normalized_reconstruction_quality: float
    reconstruction_loss: float
    score: float


@dataclass(frozen=True, slots=True)
class OptunaTuningConfig:
    n_trials: int = 12
    trial_epochs: int = 50
    reference_reconstruction_loss: float = 0.08
    input_layer: str = "log_normalized"
    covariate_keys: tuple[str, ...] = ("iPSC_line", "batch", "round", "time_point")
    control_treatment: str = "control"
    n_de_genes: int = 100
    cell_type_key: str = "time_point"
    device: str = "cpu"
    random_seed: int = 42
    study_name: str = "regenai_pt_combined_tuning"
    storage: str | None = None

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.trial_epochs < 20:
            raise ValueError("trial_epochs must be at least 20")
        if self.reference_reconstruction_loss <= 0.0:
            raise ValueError("reference_reconstruction_loss must be positive")


def normalized_reconstruction_quality(
    reconstruction_loss: float,
    reference_reconstruction_loss: float,
) -> float:
    """Convert a lower-is-better loss into a bounded higher-is-better quality."""
    if reference_reconstruction_loss <= 0.0:
        raise ValueError("reference_reconstruction_loss must be positive")
    if not np.isfinite(reconstruction_loss) or reconstruction_loss < 0.0:
        raise ValueError("reconstruction_loss must be finite and non-negative")
    return float(
        np.clip(
            1.0 - reconstruction_loss / reference_reconstruction_loss,
            0.0,
            1.0,
        )
    )


def compute_validation_score(
    *,
    r2_de: float,
    r2_mean: float,
    perturbation_fidelity: float,
    reconstruction_loss: float,
    reference_reconstruction_loss: float,
    epoch: int = 0,
) -> ValidationScore:
    """Compute the agreed higher-is-better Optuna validation objective."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    values = (r2_de, r2_mean, perturbation_fidelity)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("R-squared and perturbation-fidelity metrics must be finite")

    clipped_r2_de = float(np.clip(r2_de, -1.0, 1.0))
    clipped_r2_mean = float(np.clip(r2_mean, -1.0, 1.0))
    clipped_fidelity = float(np.clip(perturbation_fidelity, -1.0, 1.0))
    reconstruction_quality = normalized_reconstruction_quality(
        reconstruction_loss,
        reference_reconstruction_loss,
    )
    score = (
        0.4 * clipped_r2_de
        + 0.3 * clipped_r2_mean
        + 0.2 * clipped_fidelity
        + 0.1 * reconstruction_quality
    )
    return ValidationScore(
        epoch=epoch,
        r2_de=clipped_r2_de,
        r2_mean=clipped_r2_mean,
        perturbation_fidelity=clipped_fidelity,
        normalized_reconstruction_quality=reconstruction_quality,
        reconstruction_loss=float(reconstruction_loss),
        score=float(score),
    )


def best_history_validation_score(
    history: dict[str, list[float]],
    *,
    reference_reconstruction_loss: float,
    min_epoch: int = 1,
) -> ValidationScore:
    """Return the highest combined validation score after a minimum epoch."""
    if min_epoch <= 0:
        raise ValueError("min_epoch must be positive")
    required = {
        "val_mean_DE",
        "val_mean",
        "val_perturbation_fidelity",
        "val_reconstruction_loss",
    }
    missing = sorted(required - history.keys())
    if missing:
        raise KeyError(f"Training history is missing required metrics: {', '.join(missing)}")

    n_epochs = min(len(history[key]) for key in required)
    candidates: list[ValidationScore] = []
    for index in range(n_epochs):
        epoch = index + 1
        if epoch < min_epoch:
            continue
        try:
            candidate = compute_validation_score(
                r2_de=float(history["val_mean_DE"][index]),
                r2_mean=float(history["val_mean"][index]),
                perturbation_fidelity=float(history["val_perturbation_fidelity"][index]),
                reconstruction_loss=float(history["val_reconstruction_loss"][index]),
                reference_reconstruction_loss=reference_reconstruction_loss,
                epoch=epoch,
            )
        except ValueError:
            continue
        candidates.append(candidate)

    if not candidates:
        raise ValueError("No finite validation-score epochs were available")
    return max(candidates, key=lambda candidate: candidate.score)


def combined_round_validation_score(round1: ValidationScore, round2: ValidationScore) -> float:
    """Weight the two forward-transition rounds equally."""
    return float(0.5 * round1.score + 0.5 * round2.score)


def require_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Optuna tuning is optional. Install it with `pip install -e '.[tuning]'` "
            "or `pip install optuna`."
        ) from exc
    return optuna


def _sample_trial_config(trial: Any, tuning: OptunaTuningConfig) -> dict[str, Any]:
    max_warmup = min(10, max(3, tuning.trial_epochs // 4))
    max_ramp = min(60, max(20, tuning.trial_epochs - 5))
    return {
        "input_layer": tuning.input_layer,
        "n_latent": trial.suggest_categorical("n_latent", [32, 64, 128]),
        "n_hidden": trial.suggest_categorical("n_hidden", [128, 256, 512]),
        "n_layers": trial.suggest_int("n_layers", 2, 3),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 3, max_warmup),
        "ramp_epochs": trial.suggest_int("ramp_epochs", 20, max_ramp, step=5),
        "max_adversarial_weight": trial.suggest_float(
            "max_adversarial_weight",
            0.002,
            0.02,
            log=True,
        ),
        "max_epochs": tuning.trial_epochs,
        "covariate_keys": tuning.covariate_keys,
        "control_treatment": tuning.control_treatment,
        "n_de_genes": tuning.n_de_genes,
        "cell_type_key": tuning.cell_type_key,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 8,
        "early_stopping_patience": max(15, tuning.trial_epochs // 2),
        "gradient_clip_norm": 5.0,
        "reconstruction_loss": "mse",
        "random_seed": tuning.random_seed,
        "device": tuning.device,
    }


def _train_round(
    adata: ad.AnnData,
    *,
    round_number: int,
    config_payload: dict[str, Any],
) -> Any:
    import torch

    from .models.regenai_pt_config import RegenAIPTConfig
    from .models.regenai_pt_data import prepare_round_specific_adata
    from .models.regenai_pt_trainer import RegenAIPTTrainer

    base_config = RegenAIPTConfig(**config_payload)
    covariate_keys = base_config.covariate_keys
    if round_number == 2:
        covariate_keys = tuple(
            dict.fromkeys((*covariate_keys, base_config.round1_treatment_key))
        )
    effective_payload = {
        **config_payload,
        "treatment_key": "treatment",
        "dose_key": "dose",
        "covariate_keys": covariate_keys,
    }
    effective_config = RegenAIPTConfig(**effective_payload)
    prepared = prepare_round_specific_adata(
        adata,
        round_number=round_number,
        treatment_key="treatment",
        dose_key="dose",
        config=base_config,
    )
    seed = int(effective_config.random_seed + round_number - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = RegenAIPTTrainer(effective_config).fit(prepared)
    del prepared
    gc.collect()
    return trainer


def run_optuna_tuning(
    adata: ad.AnnData,
    output_dir: str | Path,
    tuning: OptunaTuningConfig,
) -> dict[str, Any]:
    """Run a persistent combined-round RegenAI-PT hyperparameter study."""
    optuna = require_optuna()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    storage = tuning.storage or f"sqlite:///{(destination / 'optuna_study.db').resolve()}"

    sampler = optuna.samplers.TPESampler(seed=tuning.random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=tuning.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        payload = _sample_trial_config(trial, tuning)
        min_epoch = int(payload["warmup_epochs"]) + 1

        round1_trainer = _train_round(
            adata,
            round_number=1,
            config_payload=payload,
        )
        round1_score = best_history_validation_score(
            round1_trainer.history,
            reference_reconstruction_loss=tuning.reference_reconstruction_loss,
            min_epoch=min_epoch,
        )
        trial.set_user_attr("round1_validation", asdict(round1_score))
        trial.report(round1_score.score, step=0)
        del round1_trainer
        gc.collect()
        if trial.should_prune():
            raise optuna.TrialPruned("Round 1 validation score was pruned")

        round2_trainer = _train_round(
            adata,
            round_number=2,
            config_payload=payload,
        )
        round2_score = best_history_validation_score(
            round2_trainer.history,
            reference_reconstruction_loss=tuning.reference_reconstruction_loss,
            min_epoch=min_epoch,
        )
        combined_score = combined_round_validation_score(round1_score, round2_score)
        trial.set_user_attr("round2_validation", asdict(round2_score))
        trial.set_user_attr("combined_validation_score", combined_score)
        trial.report(combined_score, step=1)
        del round2_trainer
        gc.collect()
        if trial.should_prune():
            raise optuna.TrialPruned("Combined validation score was pruned")
        return combined_score

    study.optimize(
        objective,
        n_trials=tuning.n_trials,
        n_jobs=1,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
    )

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError("Optuna study completed without a successful trial")

    trials_path = destination / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    best_payload = {
        "study_name": study.study_name,
        "direction": "maximize",
        "best_trial_number": int(study.best_trial.number),
        "best_score": float(study.best_value),
        "best_params": dict(study.best_params),
        "round1_validation": study.best_trial.user_attrs.get("round1_validation"),
        "round2_validation": study.best_trial.user_attrs.get("round2_validation"),
        "objective": {
            "r2_de_weight": 0.4,
            "r2_mean_weight": 0.3,
            "perturbation_fidelity_weight": 0.2,
            "normalized_reconstruction_quality_weight": 0.1,
            "reference_reconstruction_loss": tuning.reference_reconstruction_loss,
        },
        "fixed_tuning_config": asdict(tuning),
        "storage": storage,
    }
    best_path = destination / "best_hyperparameters.json"
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "best_hyperparameters_path": str(best_path),
        "trials_path": str(trials_path),
        "storage": storage,
        **best_payload,
    }


__all__ = [
    "OptunaTuningConfig",
    "ValidationScore",
    "best_history_validation_score",
    "combined_round_validation_score",
    "compute_validation_score",
    "normalized_reconstruction_quality",
    "require_optuna",
    "run_optuna_tuning",
]

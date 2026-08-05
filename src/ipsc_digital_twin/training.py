from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .forward_model import ForwardTransitionModel, build_transition_table
from .model_registry import ModelRegistry


def _build_condition_table(adata: ad.AnnData) -> pd.DataFrame:
    required = [
        'time_point',
        'round1_treatment',
        'round2_treatment',
        'iPSC_line',
        'replicate',
        'target_marker_score',
        'stress_score',
        'off_target_score',
    ]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns for baseline training: {', '.join(missing)}")

    table = (
        adata.obs[required]
        .groupby(['time_point', 'round1_treatment', 'round2_treatment', 'iPSC_line', 'replicate'], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    return table



def _train_baseline_artifact(condition_table: pd.DataFrame) -> dict[str, Any]:
    round1 = (
        condition_table[condition_table['time_point'] == 'post_round1']
        .groupby(['round1_treatment', 'iPSC_line'], dropna=False)[['target_marker_score', 'stress_score', 'off_target_score']]
        .mean()
        .reset_index()
    )
    round2 = (
        condition_table[condition_table['time_point'] == 'post_round2']
        .groupby(['round1_treatment', 'round2_treatment', 'iPSC_line'], dropna=False)[
            ['target_marker_score', 'stress_score', 'off_target_score']
        ]
        .mean()
        .reset_index()
    )
    return {
        'round1_table': round1.to_dict(orient='records'),
        'round2_table': round2.to_dict(orient='records'),
    }



def _compute_baseline_metrics(condition_table: pd.DataFrame) -> dict[str, float]:
    return {
        'n_condition_rows': float(len(condition_table)),
        'mean_target_marker_score': float(condition_table['target_marker_score'].mean()),
        'mean_stress_score': float(condition_table['stress_score'].mean()),
        'mean_off_target_score': float(condition_table['off_target_score'].mean()),
    }



def _compute_forward_transition_metrics(model: ForwardTransitionModel, transition_table: pd.DataFrame) -> dict[str, float]:
    round1_rows = transition_table[transition_table['transition_stage'] == 'round1']
    round2_rows = transition_table[transition_table['transition_stage'] == 'round2']
    metrics: dict[str, float] = {
        'n_transition_rows': float(len(transition_table)),
        'n_round1_transitions': float(len(round1_rows)),
        'n_round2_transitions': float(len(round2_rows)),
        'n_state_features': float(len(model.state_features)),
    }
    for feature in model.state_features:
        if f'{feature}_future' in transition_table.columns and f'{feature}_current' in transition_table.columns:
            metrics[f'mean_abs_delta_{feature}'] = float(
                (transition_table[f'{feature}_future'] - transition_table[f'{feature}_current']).abs().mean()
            )
    return metrics



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))



def _write_model_card(path: Path, model_record: dict[str, Any], metrics: dict[str, Any], notes: list[str]) -> None:
    md = "\n".join(
        [
            '# Model Card',
            '',
            f"- Model ID: {model_record['model_id']}",
            f"- Model Type: {model_record['model_type']}",
            f"- Dataset Version: {model_record['dataset_version']}",
            f"- Random Seed: {model_record['random_seed']}",
            f"- Artifact Hash: {model_record['artifact_hash_sha256']}",
            f"- Git Commit: {model_record.get('git_commit')}",
            '',
            '## Metrics',
            *(f'- {k}: {v}' for k, v in metrics.items()),
            '',
            '## Notes',
            *[f'- {note}' for note in notes],
        ]
    )
    path.write_text(md)


REGENAI_PT_LIMITATIONS = [
    'RegenAI-PT is a custom implementation and is not the official CPA package.',
    'It is a research model for experimental planning.',
    'It does not prove causal sufficiency.',
    'It requires prospective validation.',
    'scRNA-seq is destructive, so transitions are learned from population-level observations.',
    'It is not a clinical system.',
]



def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)



def _regenai_pt_architecture_summary(config_payload: dict[str, Any]) -> list[str]:
    return [
        f"Encoder: {config_payload.get('n_layers', 'unknown')}-layer MLP with hidden width {config_payload.get('n_hidden', 'unknown')} mapping expression to a {config_payload.get('n_latent', 'unknown')}-dimensional basal latent state.",
        'Perturbation module: learned treatment embedding plus treatment-specific dose-response scaling.',
        'Covariates: additive embeddings for configured covariates summed into the latent transition state.',
        'Decoder: MLP from perturbed latent state back to predicted expression.',
        'Adversarial heads: treatment and covariate classifiers attached to the basal latent state via gradient reversal.',
    ]



def _write_regenai_pt_model_card(
    path: Path,
    model_record: dict[str, Any],
    *,
    dataset_metadata: dict[str, Any] | None,
    config_payload: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, list[float]] | None = None,
    leakage_metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> None:
    dataset_metadata = dataset_metadata or {}
    history = history or {}
    notes = notes or []
    training_dataset_id = dataset_metadata.get('dataset_id') or model_record.get('training_config', {}).get('dataset_id', 'unknown')
    validation_losses: list[str] = []
    if history.get('val_total_loss'):
        validation_losses.append(f"- Final validation total loss: {_format_metric_value(history['val_total_loss'][-1])}")
    if history.get('val_reconstruction_loss'):
        validation_losses.append(f"- Final validation reconstruction loss: {_format_metric_value(history['val_reconstruction_loss'][-1])}")
    if 'best_val_reconstruction_loss' in metrics:
        validation_losses.append(f"- Best validation reconstruction loss: {_format_metric_value(metrics['best_val_reconstruction_loss'])}")
    if 'best_epoch' in metrics:
        validation_losses.append(f"- Restored best epoch: {metrics['best_epoch']}")
    if 'stopped_early' in metrics:
        validation_losses.append(f"- Stopped early: {metrics['stopped_early']}")
    if not validation_losses:
        validation_losses.append('- Validation losses: not available')

    reconstruction_lines = []
    for key in [
        'best_val_reconstruction_loss',
        'final_val_reconstruction_loss',
        'final_train_reconstruction_loss',
        'reconstruction_mse',
        'reconstruction_correlation',
    ]:
        if key in metrics:
            reconstruction_lines.append(f"- {key}: {_format_metric_value(metrics[key])}")
    if not reconstruction_lines:
        reconstruction_lines.append('- Reconstruction metrics not available in this training run.')

    leakage_lines = []
    leakage_metrics = leakage_metrics or {
        key: value
        for key, value in metrics.items()
        if 'leakage' in key or key.startswith('treatment_adv') or key.startswith('covariate_adv')
    }
    if leakage_metrics:
        for key, value in leakage_metrics.items():
            leakage_lines.append(f"- {key}: {_format_metric_value(value)}")
    else:
        leakage_lines.append('- Adversarial leakage metrics were not computed for this run.')

    card_lines = [
        '# Model Card',
        '',
        '## Summary',
        f"- Model type: {model_record['model_type']}",
        f"- Training dataset ID: {training_dataset_id}",
        f"- Dataset version: {model_record['dataset_version']}",
        f"- Schema version: {dataset_metadata.get('schema_version', 'unknown')}",
        f"- Preprocessing version: {dataset_metadata.get('preprocessing_version', 'unknown')}",
        f"- Random seed: {model_record['random_seed']}",
        f"- Artifact hash: {model_record['artifact_hash_sha256']}",
        f"- Git commit: {model_record.get('git_commit')}",
        '',
        '## Data Interface',
        f"- Treatment key: {config_payload.get('treatment_key', 'unknown')}",
        f"- Dose key: {config_payload.get('dose_key', 'unknown')}",
        f"- Covariate keys: {', '.join(config_payload.get('covariate_keys', [])) if config_payload.get('covariate_keys') else 'none'}",
        f"- Input layer: {config_payload.get('input_layer', 'unknown')}",
        '',
        '## Architecture Summary',
        *[f'- {line}' for line in _regenai_pt_architecture_summary(config_payload)],
        f"- n_latent: {config_payload.get('n_latent', 'unknown')}",
        '',
        '## Loss Weights',
        f"- reconstruction_loss: {config_payload.get('reconstruction_loss', 'unknown')}",
        f"- adversarial warm-up epochs: {config_payload.get('warmup_epochs', 'unknown')}",
        f"- adversarial ramp epochs: {config_payload.get('ramp_epochs', 'unknown')}",
        f"- maximum adversarial weight: {config_payload.get('max_adversarial_weight', 'unknown')}",
        f"- covariate_adversarial_weight: {config_payload.get('covariate_adversarial_weight', 'unknown')}",
        f"- perturbation_adversarial_weight: {config_payload.get('perturbation_adversarial_weight', 'unknown')}",
        f"- embedding_l2_weight: {config_payload.get('embedding_l2_weight', 'unknown')}",
        f"- dose_regularization_weight: {config_payload.get('dose_regularization_weight', 'unknown')}",
        f"- weight_decay: {config_payload.get('weight_decay', 'unknown')}",
        "- lr_scheduler: ReduceLROnPlateau(val_reconstruction_loss)",
        f"- lr_scheduler_factor: {config_payload.get('lr_scheduler_factor', 'unknown')}",
        f"- lr_scheduler_patience: {config_payload.get('lr_scheduler_patience', 'unknown')}",
        f"- early_stopping_patience: {config_payload.get('early_stopping_patience', 'unknown')}",
        f"- gradient_clip_norm: {config_payload.get('gradient_clip_norm', 'unknown')}",
        '',
        '## Training Run',
        f"- Training epochs: {metrics.get('epochs_trained', metrics.get('round1_epochs_trained', 'unknown'))}",
        f"- Max epochs configured: {config_payload.get('max_epochs', 'unknown')}",
        *validation_losses,
        '',
        '## Reconstruction Metrics',
        *reconstruction_lines,
        '',
        '## Adversarial Leakage Metrics',
        *leakage_lines,
        '',
        '## Notes',
        *[f'- {note}' for note in notes],
        '',
        '## Known Limitations',
        *[f'- {item}' for item in REGENAI_PT_LIMITATIONS],
    ]
    path.write_text("\n".join(card_lines) + "\n", encoding='utf-8')



def _history_rows(history: dict[str, list[float]], phase: str) -> list[dict[str, Any]]:
    n_rows = max((len(values) for values in history.values()), default=0)
    rows: list[dict[str, Any]] = []
    for epoch_idx in range(n_rows):
        row: dict[str, Any] = {'phase': phase, 'epoch': epoch_idx + 1}
        for key, values in history.items():
            row[key] = values[epoch_idx] if epoch_idx < len(values) else None
        rows.append(row)
    return rows


def _final_loss_metrics(history: dict[str, list[float]], prefix: str = '') -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split in ('train', 'val'):
        for component in (
            'reconstruction_loss',
            'treatment_adv_loss',
            'covariate_adv_loss',
            'embedding_l2_loss',
            'dose_regularization_loss',
            'total_loss',
        ):
            key = f'{split}_{component}'
            values = history.get(key, [])
            if values:
                metrics[f'{prefix}final_{key}'] = float(values[-1])
    if history.get('adversarial_weight'):
        metrics[f'{prefix}final_adversarial_weight'] = float(history['adversarial_weight'][-1])
    if history.get('learning_rate'):
        metrics[f'{prefix}final_learning_rate'] = float(history['learning_rate'][-1])
    return metrics



def _save_regenai_pt_supporting_artifacts(
    output_dir: Path,
    config_payload: dict[str, Any],
    mappings_payload: dict[str, Any],
    history_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    prefix: str,
) -> dict[str, str]:
    config_path = output_dir / f'{prefix}_config.json'
    mappings_path = output_dir / f'{prefix}_mappings.json'
    history_path = output_dir / f'{prefix}_training_history.csv'
    metrics_path = output_dir / f'{prefix}_metrics.json'

    _write_json(config_path, config_payload)
    _write_json(mappings_path, mappings_payload)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    _write_json(metrics_path, metrics)

    return {
        'config_path': str(config_path),
        'mappings_path': str(mappings_path),
        'history_path': str(history_path),
        'metrics_path': str(metrics_path),
    }



def train_and_register_baseline(
    dataset_path: str | Path,
    dataset_id: str,
    dataset_version: int | str,
    output_dir: str | Path,
    model_registry: ModelRegistry,
    random_seed: int = 0,
) -> dict[str, Any]:
    np.random.seed(random_seed)

    adata = ad.read_h5ad(Path(dataset_path))
    condition_table = _build_condition_table(adata)
    artifact_data = _train_baseline_artifact(condition_table)
    metrics = _compute_baseline_metrics(condition_table)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / f'{dataset_id}_baseline_model.pkl'
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact_data, f)

    metrics_path = out_dir / f'{dataset_id}_baseline_metrics.json'
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2))

    record = model_registry.register_model_artifact(
        artifact_path=artifact_path,
        model_type='baseline',
        dataset_version=dataset_version,
        training_config={'dataset_id': dataset_id, 'model_type': 'baseline'},
        metrics=metrics,
        random_seed=random_seed,
    )

    card_path = out_dir / f"{record['model_id']}_model_card.md"
    _write_model_card(
        card_path,
        record,
        metrics,
        notes=['Legacy condition-level statistical model preserved for backwards compatibility.'],
    )

    return {
        'model_record': record,
        'artifact_path': str(artifact_path),
        'metrics_path': str(metrics_path),
        'model_card_path': str(card_path),
    }



def train_and_register_forward_transition(
    dataset_path: str | Path,
    dataset_id: str,
    dataset_version: int | str,
    output_dir: str | Path,
    model_registry: ModelRegistry,
    random_seed: int = 0,
) -> dict[str, Any]:
    np.random.seed(random_seed)

    adata = ad.read_h5ad(Path(dataset_path))
    transition_table = build_transition_table(adata)
    model = ForwardTransitionModel().fit_transition_table(transition_table)
    metrics = _compute_forward_transition_metrics(model, transition_table)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / f'{dataset_id}_forward_transition_model.pkl'
    model.save(artifact_path)

    metrics_path = out_dir / f'{dataset_id}_forward_transition_metrics.json'
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2))

    record = model_registry.register_model_artifact(
        artifact_path=artifact_path,
        model_type='forward_transition',
        dataset_version=dataset_version,
        training_config={
            'dataset_id': dataset_id,
            'model_type': 'forward_transition',
            'transition_model': 'baseline',
            'primary_prediction_target': 'future_population_state',
        },
        metrics=metrics,
        random_seed=random_seed,
    )

    card_path = out_dir / f"{record['model_id']}_model_card.md"
    _write_model_card(
        card_path,
        record,
        metrics,
        notes=[
            'Primary target is future population state, not final_score.',
            'Transitions are matched at population level because scRNA-seq is destructive and same-cell pairing is unavailable.',
            'Recommendations from this model remain experimental and require prospective validation.',
        ],
    )

    return {
        'model_record': record,
        'artifact_path': str(artifact_path),
        'metrics_path': str(metrics_path),
        'model_card_path': str(card_path),
    }



def train_and_register_regenai_pt(
    dataset_path: str | Path,
    dataset_id: str,
    dataset_version: int | str,
    output_dir: str | Path,
    model_registry: ModelRegistry,
    config_kwargs: dict[str, Any],
    treatment_mode: str = 'single_round',
    random_seed: int = 0,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from .models.regenai_pt_config import RegenAIPTConfig
    from .models.regenai_pt_data import prepare_round_specific_adata
    from .models.regenai_pt_trainer import RegenAIPTTrainer

    np.random.seed(random_seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(Path(dataset_path))
    config = RegenAIPTConfig(**config_kwargs)
    prefix = f'{dataset_id}_regenai_pt'

    if treatment_mode == 'combined_rounds':
        adapter = RegenAIPTForwardAdapter().fit(adata, config)
        artifact_path = out_dir / f'{prefix}_combined_model.pt'
        adapter.save(artifact_path)

        round1_mappings = None if adapter.round1_trainer is None or adapter.round1_trainer.mappings is None else asdict(adapter.round1_trainer.mappings)
        round2_mappings = None if adapter.round2_trainer is None or adapter.round2_trainer.mappings is None else asdict(adapter.round2_trainer.mappings)
        history_rows = []
        if adapter.round1_trainer is not None:
            history_rows.extend(_history_rows(adapter.round1_trainer.history, 'round1'))
        if adapter.round2_trainer is not None:
            history_rows.extend(_history_rows(adapter.round2_trainer.history, 'round2'))

        metrics = {
            'treatment_mode': 'combined_rounds',
            'n_cells': float(adata.n_obs),
            'n_genes': float(adata.n_vars),
            'round1_epochs_trained': float(adapter.round1_trainer.epochs_trained if adapter.round1_trainer is not None else 0),
            'round2_epochs_trained': float(adapter.round2_trainer.epochs_trained if adapter.round2_trainer is not None else 0),
            'round1_best_val_reconstruction_loss': float(adapter.round1_trainer.best_val_reconstruction_loss or 0.0) if adapter.round1_trainer is not None else 0.0,
            'round2_best_val_reconstruction_loss': float(adapter.round2_trainer.best_val_reconstruction_loss or 0.0) if adapter.round2_trainer is not None else 0.0,
            'round1_best_epoch': float(adapter.round1_trainer.best_epoch or 0) if adapter.round1_trainer is not None else 0.0,
            'round2_best_epoch': float(adapter.round2_trainer.best_epoch or 0) if adapter.round2_trainer is not None else 0.0,
            'round1_stopped_early': adapter.round1_trainer.stopped_early if adapter.round1_trainer is not None else False,
            'round2_stopped_early': adapter.round2_trainer.stopped_early if adapter.round2_trainer is not None else False,
        }
        if adapter.round1_trainer is not None:
            metrics.update(_final_loss_metrics(adapter.round1_trainer.history, prefix='round1_'))
        if adapter.round2_trainer is not None:
            metrics.update(_final_loss_metrics(adapter.round2_trainer.history, prefix='round2_'))
        extra_paths = _save_regenai_pt_supporting_artifacts(
            output_dir=out_dir,
            config_payload={
                'treatment_mode': treatment_mode,
                'adapter_config': asdict(config),
            },
            mappings_payload={
                'round1_mappings': round1_mappings,
                'round2_mappings': round2_mappings,
            },
            history_rows=history_rows,
            metrics=metrics,
            prefix=prefix,
        )
        record = model_registry.register_model_artifact(
            artifact_path=artifact_path,
            model_type='regenai_pt',
            dataset_version=dataset_version,
            training_config={
                'dataset_id': dataset_id,
                'model_type': 'regenai_pt',
                'transition_model': 'regenai_pt',
                'treatment_mode': treatment_mode,
                'config': asdict(config),
            },
            metrics=metrics,
            random_seed=random_seed,
        )
        card_path = out_dir / f"{record['model_id']}_model_card.md"
        _write_regenai_pt_model_card(
            card_path,
            record,
            dataset_metadata=dataset_metadata,
            config_payload=asdict(config),
            metrics=metrics,
            history={
                'round1_val_reconstruction_loss': adapter.round1_trainer.history.get('val_reconstruction_loss', []) if adapter.round1_trainer is not None else [],
                'round2_val_reconstruction_loss': adapter.round2_trainer.history.get('val_reconstruction_loss', []) if adapter.round2_trainer is not None else [],
            },
            notes=[
                'Combined round training saves a round1 and round2 forward simulator bundle.',
                'Primary target is future latent or expression state, not final_score.',
                'Recommendations remain experimental and require prospective validation.',
            ],
        )
        return {
            'model_record': record,
            'artifact_path': str(artifact_path),
            'model_card_path': str(card_path),
            **extra_paths,
        }

    treatment_key = config.treatment_key
    if treatment_key == config.round1_treatment_key:
        train_adata = prepare_round_specific_adata(
            adata,
            round_number=1,
            treatment_key='treatment',
            dose_key='dose',
            config=config,
        )
        role = 'round1'
    elif treatment_key == config.round2_treatment_key:
        train_adata = prepare_round_specific_adata(
            adata,
            round_number=2,
            treatment_key='treatment',
            dose_key='dose',
            config=config,
        )
        role = 'round2'
    else:
        train_adata = adata.copy()
        role = 'custom'

    effective_config = RegenAIPTConfig(**{**asdict(config), 'treatment_key': 'treatment', 'dose_key': 'dose'})
    trainer = RegenAIPTTrainer(effective_config).fit(train_adata)
    artifact_path = out_dir / f'{prefix}_{role}_model.pt'
    trainer.save(artifact_path)

    mappings_payload = {} if trainer.mappings is None else asdict(trainer.mappings)
    metrics = {
        'treatment_mode': treatment_mode,
        'training_role': role,
        'n_cells': float(train_adata.n_obs),
        'n_genes': float(train_adata.n_vars),
        'epochs_trained': float(trainer.epochs_trained),
        'best_val_reconstruction_loss': float(trainer.best_val_reconstruction_loss or 0.0),
        'best_epoch': float(trainer.best_epoch or 0),
        'stopped_early': trainer.stopped_early,
        'final_train_total_loss': float(trainer.history['train_total_loss'][-1]) if trainer.history['train_total_loss'] else 0.0,
        'final_val_total_loss': float(trainer.history['val_total_loss'][-1]) if trainer.history['val_total_loss'] else 0.0,
        'final_train_reconstruction_loss': float(trainer.history['train_reconstruction_loss'][-1]) if trainer.history['train_reconstruction_loss'] else 0.0,
        'final_val_reconstruction_loss': float(trainer.history['val_reconstruction_loss'][-1]) if trainer.history['val_reconstruction_loss'] else 0.0,
        'n_treatments': float(len(trainer.mappings.treatment_to_id) if trainer.mappings is not None else 0),
    }
    metrics.update(_final_loss_metrics(trainer.history))
    extra_paths = _save_regenai_pt_supporting_artifacts(
        output_dir=out_dir,
        config_payload={
            'treatment_mode': treatment_mode,
            'requested_config': asdict(config),
            'effective_training_config': asdict(effective_config),
        },
        mappings_payload=mappings_payload,
        history_rows=_history_rows(trainer.history, role),
        metrics=metrics,
        prefix=prefix,
    )
    record = model_registry.register_model_artifact(
        artifact_path=artifact_path,
        model_type='regenai_pt',
        dataset_version=dataset_version,
        training_config={
            'dataset_id': dataset_id,
            'model_type': 'regenai_pt',
            'transition_model': 'regenai_pt',
            'treatment_mode': treatment_mode,
            'training_role': role,
            'config': asdict(config),
            'effective_training_config': asdict(effective_config),
        },
        metrics=metrics,
        random_seed=random_seed,
    )
    card_path = out_dir / f"{record['model_id']}_model_card.md"
    _write_regenai_pt_model_card(
        card_path,
        record,
        dataset_metadata=dataset_metadata,
        config_payload=asdict(config),
        metrics=metrics,
        history=trainer.history,
        notes=[
            f'Single-role training prepared data for {role} treatment modeling.',
            'Primary target is future latent or expression state, not final_score.',
            'Recommendations remain experimental and require prospective validation.',
        ],
    )
    return {
        'model_record': record,
        'artifact_path': str(artifact_path),
        'model_card_path': str(card_path),
        **extra_paths,
    }



def train_and_register_regenai_pt_forward_transition(
    dataset_path: str | Path,
    dataset_id: str,
    dataset_version: int | str,
    output_dir: str | Path,
    model_registry: ModelRegistry,
    random_seed: int = 0,
    max_epochs: int = 5,
    batch_size: int = 32,
    device: str = 'auto',
    input_layer: str = 'raw_counts',
    warmup_epochs: int = 20,
    ramp_epochs: int = 20,
    max_adversarial_weight: float = 0.05,
    gradient_clip_norm: float = 5.0,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 5,
    early_stopping_patience: int = 15,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return train_and_register_regenai_pt(
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        output_dir=output_dir,
        model_registry=model_registry,
        config_kwargs={
            'input_layer': input_layer,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'random_seed': random_seed,
            'device': device,
            'warmup_epochs': warmup_epochs,
            'ramp_epochs': ramp_epochs,
            'max_adversarial_weight': max_adversarial_weight,
            'gradient_clip_norm': gradient_clip_norm,
            'lr_scheduler_factor': lr_scheduler_factor,
            'lr_scheduler_patience': lr_scheduler_patience,
            'early_stopping_patience': early_stopping_patience,
        },
        treatment_mode='combined_rounds',
        random_seed=random_seed,
        dataset_metadata=dataset_metadata,
    )

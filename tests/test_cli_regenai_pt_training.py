from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.cli import main
from ipsc_digital_twin.model_registry import ModelRegistry
from ipsc_digital_twin.registry import DatasetRegistry


def _make_round_trainable_adata(path: Path, n_per_group: int = 6) -> None:
    rng = np.random.default_rng(21)
    rows = []
    for treatment in ['control', 'A', 'B']:
        for i in range(n_per_group):
            rows.append(
                {
                    'time_point': 'intermediate',
                    'round': 0,
                    'round1_treatment': treatment,
                    'round1_dose': 0.0 if treatment == 'control' else (1.0 if treatment == 'A' else 2.0),
                    'round2_treatment': 'control',
                    'round2_dose': 0.0,
                    'treatment_sequence': f'{treatment}->control',
                    'iPSC_line': f'line{(i % 2) + 1}',
                    'batch': 'batch1',
                    'replicate': f'r{(i % 3) + 1}',
                    'sequencing_run': f'run{(i % 2) + 1}',
                    'target_marker_score': 0.2,
                    'stress_score': 0.05,
                    'off_target_score': 0.04,
                    'pseudotime': 0.2,
                    'fate_probability': 0.25,
                }
            )
            for round2_treatment in ['A', 'B']:
                rows.append(
                    {
                        'time_point': 'post_round1',
                        'round': 1,
                        'round1_treatment': treatment,
                        'round1_dose': 0.0 if treatment == 'control' else (1.0 if treatment == 'A' else 2.0),
                        'round2_treatment': round2_treatment,
                        'round2_dose': 0.5 if round2_treatment == 'A' else 1.5,
                        'treatment_sequence': f'{treatment}->{round2_treatment}',
                        'iPSC_line': f'line{(i % 2) + 1}',
                        'batch': 'batch1',
                        'replicate': f'r{(i % 3) + 1}',
                        'sequencing_run': f'run{(i % 2) + 1}',
                        'target_marker_score': 0.45,
                        'stress_score': 0.08,
                        'off_target_score': 0.05,
                        'pseudotime': 0.45,
                        'fate_probability': 0.5,
                    }
                )
                rows.append(
                    {
                        'time_point': 'post_round2',
                        'round': 2,
                        'round1_treatment': treatment,
                        'round1_dose': 0.0 if treatment == 'control' else (1.0 if treatment == 'A' else 2.0),
                        'round2_treatment': round2_treatment,
                        'round2_dose': 0.5 if round2_treatment == 'A' else 1.5,
                        'treatment_sequence': f'{treatment}->{round2_treatment}',
                        'iPSC_line': f'line{(i % 2) + 1}',
                        'batch': 'batch1',
                        'replicate': f'r{(i % 3) + 1}',
                        'sequencing_run': f'run{(i % 2) + 1}',
                        'target_marker_score': 0.65,
                        'stress_score': 0.1,
                        'off_target_score': 0.06,
                        'pseudotime': 0.75,
                        'fate_probability': 0.72,
                    }
                )
    obs = pd.DataFrame(rows)
    x = rng.poisson(2.0, size=(len(obs), 10)).astype(float)
    var = pd.DataFrame(index=[f'g{i}' for i in range(10)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers['raw_counts'] = x.copy()
    adata.layers['log_normalized'] = np.log1p(x)
    adata.write_h5ad(path)


def test_cli_regenai_pt_training_and_registry_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    registry_dir = tmp_path / 'state'
    registry_dir.mkdir(parents=True, exist_ok=True)

    ds_path = tmp_path / 'rounds.h5ad'
    _make_round_trainable_adata(ds_path)

    ds_reg = DatasetRegistry(registry_dir / 'datasets.json')
    ds_record = ds_reg.register_dataset(
        anndata_path=ds_path,
        experiment_id='exp_regenai_pt',
        preprocessing_version='p1',
        schema_version='s1',
    )

    out_dir = tmp_path / 'outputs' / 'regenai_pt_training'
    rc = main(
        [
            'train',
            '--registry-dir',
            str(registry_dir),
            '--dataset-id',
            ds_record['dataset_id'],
            '--model-type',
            'regenai_pt',
            '--output-dir',
            str(out_dir),
            '--treatment-key',
            'round1_treatment',
            '--dose-key',
            'round1_dose',
            '--covariate-keys',
            'iPSC_line,batch,round,time_point',
            '--max-epochs',
            '1',
            '--batch-size',
            '8',
            '--n-latent',
            '8',
            '--warmup-epochs',
            '0',
            '--ramp-epochs',
            '1',
            '--max-adversarial-weight',
            '0.05',
            '--device',
            'cpu',
            '--input-layer',
            'raw_counts',
        ]
    )
    assert rc == 0
    assert any(path.name.endswith('_round1_model.pt') for path in out_dir.iterdir())
    assert any(path.name.endswith('_config.json') for path in out_dir.iterdir())
    assert any(path.name.endswith('_mappings.json') for path in out_dir.iterdir())
    assert any(path.name.endswith('_training_history.csv') for path in out_dir.iterdir())
    assert any(path.name.endswith('_metrics.json') for path in out_dir.iterdir())
    assert any(path.name.endswith('_model_card.md') for path in out_dir.iterdir())

    history_path = next(out_dir.glob('*_training_history.csv'))
    history = pd.read_csv(history_path)
    for split in ('train', 'val'):
        for component in (
            'reconstruction_loss',
            'treatment_adv_loss',
            'covariate_adv_loss',
            'embedding_l2_loss',
            'dose_regularization_loss',
            'total_loss',
        ):
            assert f'{split}_{component}' in history.columns
    assert 'adversarial_weight' in history.columns
    assert history['adversarial_weight'].iloc[-1] == pytest.approx(0.05)
    assert 'learning_rate' in history.columns
    assert history['learning_rate'].iloc[-1] == pytest.approx(0.001)
    for split in ('train', 'val'):
        for metric in (
            'mean',
            'mean_DE',
            'Var',
            'Var_DE',
            'perturbation_disent',
            'cell_type_disent',
            'covariate_adv_accuracy',
            'covariate_adv_accuracy_iPSC_line',
        ):
            assert f'{split}_{metric}' in history.columns

    metrics_path = next(out_dir.glob('*_metrics.json'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert metrics['best_epoch'] == 1
    assert not bool(metrics['stopped_early'])

    reg = ModelRegistry(registry_dir / 'models.json')
    models = reg.list_models()
    assert models[-1]['model_type'] == 'regenai_pt'


def test_cli_regenai_pt_combined_training_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    registry_dir = tmp_path / 'state_combined'
    registry_dir.mkdir(parents=True, exist_ok=True)

    ds_path = tmp_path / 'rounds_combined.h5ad'
    _make_round_trainable_adata(ds_path)

    ds_reg = DatasetRegistry(registry_dir / 'datasets.json')
    ds_record = ds_reg.register_dataset(
        anndata_path=ds_path,
        experiment_id='exp_combined',
        preprocessing_version='p1',
        schema_version='s1',
    )

    out_dir = tmp_path / 'outputs' / 'regenai_pt_combined'
    rc = main(
        [
            'train',
            '--registry-dir',
            str(registry_dir),
            '--dataset-id',
            ds_record['dataset_id'],
            '--model-type',
            'regenai_pt',
            '--output-dir',
            str(out_dir),
            '--treatment-mode',
            'combined_rounds',
            '--covariate-keys',
            'iPSC_line,batch,round,time_point',
            '--max-epochs',
            '1',
            '--batch-size',
            '8',
            '--n-latent',
            '8',
            '--device',
            'cpu',
            '--input-layer',
            'raw_counts',
        ]
    )
    assert rc == 0
    assert any(path.name.endswith('_combined_model.pt') for path in out_dir.iterdir())


def test_old_baseline_cli_still_works(tmp_path: Path) -> None:
    registry_dir = tmp_path / 'registry'
    registry_dir.mkdir(parents=True, exist_ok=True)

    ds_path = tmp_path / 'baseline_train.h5ad'
    n = 30
    x = np.random.poisson(2.0, size=(n, 10)).astype(float)
    obs = pd.DataFrame(
        {
            'time_point': ['intermediate', 'post_round1', 'post_round2'] * 10,
            'round1_treatment': ['A', 'B', 'C'] * 10,
            'round2_treatment': ['X', 'Y', 'Z'] * 10,
            'iPSC_line': ['line1', 'line2', 'line3'] * 10,
            'replicate': ['r1', 'r2', 'r3'] * 10,
            'target_marker_score': np.random.rand(n),
            'stress_score': np.random.rand(n),
            'off_target_score': np.random.rand(n),
        }
    )
    ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f'g{i}' for i in range(10)])).write_h5ad(ds_path)

    ds_reg = DatasetRegistry(registry_dir / 'datasets.json')
    ds_record = ds_reg.register_dataset(
        anndata_path=ds_path,
        experiment_id='exp_base',
        preprocessing_version='p1',
        schema_version='s1',
    )
    rc = main(
        [
            'train',
            '--registry-dir',
            str(registry_dir),
            '--dataset-id',
            ds_record['dataset_id'],
            '--model-type',
            'baseline',
            '--output-dir',
            str(tmp_path / 'baseline_out'),
            '--random-seed',
            '13',
        ]
    )
    assert rc == 0

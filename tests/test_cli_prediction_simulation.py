from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.cli import main
from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig


def _make_round_adata(n_per_group: int = 6) -> ad.AnnData:
    rng = np.random.default_rng(31)
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
    return adata


def _train_and_save_adapter(tmp_path: Path) -> tuple[Path, Path]:
    adata = _make_round_adata()
    model_path = tmp_path / 'combined_model.pt'
    input_path = tmp_path / 'current.h5ad'
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    current.write_h5ad(input_path)
    adapter = RegenAIPTForwardAdapter().fit(
        adata,
        RegenAIPTConfig(
            input_layer='raw_counts',
            covariate_keys=('replicate', 'sequencing_run'),
            n_latent=4,
            n_hidden=8,
            n_layers=2,
            dropout=0.0,
            max_epochs=1,
            batch_size=8,
            device='cpu',
        ),
    )
    adapter.save(model_path)
    return model_path, input_path


def test_predict_transition_cli_works_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    model_path, input_path = _train_and_save_adapter(tmp_path)
    output_dir = tmp_path / 'prediction'
    rc = main(
        [
            'predict-transition',
            '--model-path',
            str(model_path),
            '--adata',
            str(input_path),
            '--treatment',
            'A',
            '--dose',
            '10',
            '--output-dir',
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / 'predicted_expression.npy').exists()
    assert (output_dir / 'predicted_expression.h5ad').exists()
    assert (output_dir / 'predicted_latent.csv').exists()
    assert (output_dir / 'prediction_summary.json').exists()
    assert (output_dir / 'simulation_report.md').exists()


def test_simulate_sequence_cli_works_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    model_path, input_path = _train_and_save_adapter(tmp_path)
    output_dir = tmp_path / 'sequence'
    rc = main(
        [
            'simulate-sequence',
            '--model-path',
            str(model_path),
            '--adata',
            str(input_path),
            '--round1-treatment',
            'A',
            '--round1-dose',
            '10',
            '--round2-treatment',
            'B',
            '--round2-dose',
            '5',
            '--output-dir',
            str(output_dir),
        ]
    )
    assert rc == 0
    assert (output_dir / 'predicted_expression.npy').exists()
    assert (output_dir / 'predicted_expression.h5ad').exists()
    assert (output_dir / 'predicted_latent.csv').exists()
    assert (output_dir / 'prediction_summary.json').exists()
    assert (output_dir / 'simulation_report.md').exists()


def test_invalid_treatment_gives_clear_error_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    model_path, input_path = _train_and_save_adapter(tmp_path)
    output_dir = tmp_path / 'invalid'
    rc = main(
        [
            'predict-transition',
            '--model-path',
            str(model_path),
            '--adata',
            str(input_path),
            '--treatment',
            'NOT_A_REAL_TREATMENT',
            '--dose',
            '10',
            '--output-dir',
            str(output_dir),
        ]
    )
    assert rc == 1

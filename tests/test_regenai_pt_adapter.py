from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.forward_model import (
    CurrentStateProfile,
    ForwardTransitionModel,
    TargetStateProfile,
    build_transition_table,
)
from ipsc_digital_twin.inverse_recommender import recommend_inverse_treatments
from ipsc_digital_twin.models.forward_interface import ForwardTransitionModelInterface


def _make_round_adata(n_per_group: int = 6) -> ad.AnnData:
    rng = np.random.default_rng(7)
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
                    'target_marker_score': 0.2 + 0.05 * (treatment == 'A') + 0.08 * (treatment == 'B'),
                    'stress_score': 0.05 + 0.02 * (treatment == 'B'),
                    'off_target_score': 0.04 + 0.01 * (treatment == 'control'),
                    'pseudotime': 0.2,
                    'fate_probability': 0.3,
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
                        'target_marker_score': 0.45 + 0.03 * (round2_treatment == 'B'),
                        'stress_score': 0.08 + 0.02 * (treatment == 'B'),
                        'off_target_score': 0.06 + 0.01 * (round2_treatment == 'B'),
                        'pseudotime': 0.5,
                        'fate_probability': 0.55,
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
                        'target_marker_score': 0.65 + 0.05 * (round2_treatment == 'B'),
                        'stress_score': 0.12 + 0.03 * (round2_treatment == 'B'),
                        'off_target_score': 0.07 + 0.02 * (treatment == 'B'),
                        'pseudotime': 0.8,
                        'fate_probability': 0.75,
                    }
                )
    obs = pd.DataFrame(rows)
    x = rng.poisson(2.0, size=(len(obs), 12)).astype(float)
    var = pd.DataFrame(index=[f'g{i}' for i in range(12)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers['raw_counts'] = x.copy()
    adata.layers['log_normalized'] = np.log1p(x)
    return adata


def _make_library() -> list[dict[str, object]]:
    return [
        {'treatment_name': 'control', 'active': True, 'pathway': 'baseline'},
        {'treatment_name': 'A', 'active': True, 'pathway': 'wnt'},
        {'treatment_name': 'B', 'active': True, 'pathway': 'tgfb'},
    ]


def test_forward_transition_baseline_still_works() -> None:
    adata = _make_round_adata()
    model = ForwardTransitionModel().fit_transition_table(build_transition_table(adata))
    recs = recommend_inverse_treatments(
        current_state=CurrentStateProfile(
            state_features={
                'target_marker_score': 0.2,
                'stress_score': 0.1,
                'off_target_score': 0.1,
                'pseudotime': 0.2,
                'fate_probability': 0.2,
            },
            metadata={'iPSC_line': 'line1'},
        ),
        target_state=TargetStateProfile(state_features={'target_marker_score': 0.8, 'stress_score': 0.1}),
        treatment_library=_make_library(),
        forward_model=model,
        top_n=3,
    )
    assert not recs.empty
    assert 'recommendation_score' in recs.columns


def test_regenai_pt_adapter_conforms_and_predicts_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig

    adata = _make_round_adata()
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    config = RegenAIPTConfig(
        input_layer='raw_counts',
        covariate_keys=('replicate', 'sequencing_run'),
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
        max_epochs=1,
        batch_size=8,
        device='cpu',
    )
    adapter = RegenAIPTForwardAdapter().fit(adata, config)

    assert isinstance(adapter, ForwardTransitionModelInterface)

    expr_pred = adapter.predict_expression(current, {'round1_treatment': 'A', 'dose': 1.0})
    latent_pred = adapter.predict_latent(current, {'round1_treatment': 'A', 'dose': 1.0})
    assert expr_pred['x_hat'].shape == (current.n_obs, current.n_vars)
    assert latent_pred['z_total'].shape[0] == current.n_obs

    current_profile = CurrentStateProfile(
        state_features={
            'target_marker_score': 0.2,
            'stress_score': 0.1,
            'off_target_score': 0.1,
            'pseudotime': 0.2,
            'fate_probability': 0.2,
        },
        metadata={'adata': current, 'iPSC_line': 'line1'},
    )
    pred = adapter.predict_future_state(current_profile, round1_treatment='A', round2_treatment='B', iPSC_line='line1')
    assert pred.stage == 'post_round2'
    assert set(['target_marker_score', 'stress_score', 'off_target_score']).issubset(pred.state_features)
    assert isinstance(pred.metadata['expression'], np.ndarray)

    sequence = adapter.simulate_sequence(current, 'A->B', covariates={'iPSC_line': 'line1'})
    reverse = adapter.simulate_sequence(current, 'B->A', covariates={'iPSC_line': 'line1'})
    assert sequence['x_hat'].shape == (current.n_obs, current.n_vars)
    assert not np.allclose(sequence['x_hat'], reverse['x_hat'])

    save_path = tmp_path / 'adapter.pt'
    adapter.save(save_path)
    loaded = RegenAIPTForwardAdapter.load(save_path)
    loaded_pred = loaded.predict_future_state(current_profile, round1_treatment='A', iPSC_line='line1')
    assert loaded_pred.stage == 'post_round1'


def test_inverse_recommender_accepts_regenai_pt_adapter_if_torch_available() -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig

    adata = _make_round_adata()
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
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
    current_state = CurrentStateProfile(
        state_features={
            'target_marker_score': 0.25,
            'stress_score': 0.1,
            'off_target_score': 0.1,
            'pseudotime': 0.2,
            'fate_probability': 0.3,
        },
        metadata={'adata': current, 'iPSC_line': 'line1'},
    )
    target_state = TargetStateProfile(
        state_features={
            'target_marker_score': 0.8,
            'stress_score': 0.1,
            'off_target_score': 0.1,
            'pseudotime': 0.8,
            'fate_probability': 0.8,
        }
    )
    recs = recommend_inverse_treatments(
        current_state=current_state,
        target_state=target_state,
        treatment_library=_make_library(),
        forward_model=adapter,
        top_n=5,
    )
    assert not recs.empty
    assert 'predicted_target_marker_score' in recs.columns

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.inverse_recommender import (
    RecommendationConstraint,
    recommend_inverse_treatments,
)


def _make_round_adata(n_per_group: int = 6) -> ad.AnnData:
    rng = np.random.default_rng(11)
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
                    'target_marker_score': 0.15 + 0.08 * (treatment == 'A') + 0.04 * (treatment == 'B'),
                    'stress_score': 0.05 + 0.03 * (treatment == 'B'),
                    'off_target_score': 0.04,
                    'pseudotime': 0.15,
                    'fate_probability': 0.2,
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
                        'target_marker_score': 0.35 + 0.06 * (treatment == 'A') + 0.03 * (round2_treatment == 'B'),
                        'stress_score': 0.08 + 0.04 * (treatment == 'B'),
                        'off_target_score': 0.05 + 0.01 * (round2_treatment == 'B'),
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
                        'target_marker_score': 0.55 + 0.08 * (treatment == 'A') + 0.05 * (round2_treatment == 'B'),
                        'stress_score': 0.1 + 0.04 * (round2_treatment == 'B'),
                        'off_target_score': 0.05 + 0.02 * (treatment == 'B'),
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


def _make_library() -> list[dict[str, object]]:
    return [
        {'treatment_name': 'control', 'active': True, 'pathway': 'baseline'},
        {'treatment_name': 'A', 'active': True, 'pathway': 'wnt'},
        {'treatment_name': 'B', 'active': True, 'pathway': 'tgfb'},
    ]


def _required_cols() -> set[str]:
    return {
        'recommendation_type',
        'round1_treatment',
        'round2_treatment',
        'treatment_sequence',
        'predicted_to_target_distance',
        'current_to_target_distance',
        'distance_improvement',
        'cosine_similarity_to_target',
        'predicted_target_marker_score',
        'predicted_stress_score',
        'predicted_off_target_score',
        'uncertainty',
        'recommendation_score',
    }


def test_regenai_pt_recommender_runs_one_and_two_step_if_torch_available() -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig

    adata = _make_round_adata()
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    target = adata[adata.obs['time_point'].astype(str) == 'post_round2'].copy()
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

    recs = recommend_inverse_treatments(
        current_state=current,
        target_state=target,
        treatment_library=_make_library(),
        forward_model=adapter,
        top_n=10,
    )
    assert not recs.empty
    assert _required_cols().issubset(recs.columns)
    assert (recs['recommendation_type'] == 'single_round').any()
    assert (recs['recommendation_type'] == 'sequential').any()


def test_regenai_pt_recommender_one_step_only_if_torch_available() -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig

    adata = _make_round_adata()
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    target = adata[adata.obs['time_point'].astype(str) == 'post_round1'].copy()
    adapter = RegenAIPTForwardAdapter().fit(
        adata,
        RegenAIPTConfig(input_layer='raw_counts', covariate_keys=('replicate', 'sequencing_run'), max_epochs=1, batch_size=8, device='cpu'),
    )
    recs = recommend_inverse_treatments(
        current_state=current,
        target_state=target,
        treatment_library=_make_library(),
        forward_model=adapter,
        constraints=RecommendationConstraint(include_single_round=True, max_sequences=3, fixed_round2_treatments=[]),
        top_n=3,
    )
    assert not recs.empty
    assert (recs['recommendation_type'] == 'single_round').any()


def test_sequence_order_differs_if_torch_available() -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig

    adata = _make_round_adata()
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    target = adata[adata.obs['time_point'].astype(str) == 'post_round2'].copy()
    adapter = RegenAIPTForwardAdapter().fit(
        adata,
        RegenAIPTConfig(input_layer='raw_counts', covariate_keys=('replicate', 'sequencing_run'), max_epochs=1, batch_size=8, device='cpu'),
    )
    recs = recommend_inverse_treatments(
        current_state=current,
        target_state=target,
        treatment_library=_make_library(),
        forward_model=adapter,
        constraints=RecommendationConstraint(include_single_round=False, fixed_round1_treatments=['A', 'B'], fixed_round2_treatments=['A', 'B']),
        top_n=10,
    )
    row_ab = recs.loc[recs['treatment_sequence'] == 'A->B'].iloc[0]
    row_ba = recs.loc[recs['treatment_sequence'] == 'B->A'].iloc[0]
    assert row_ab['recommendation_score'] != row_ba['recommendation_score']

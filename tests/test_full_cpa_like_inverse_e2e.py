from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.inverse_recommender import recommend_inverse_treatments


def _make_e2e_mock_adata(n_per_group: int = 4) -> ad.AnnData:
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    gene_names = [f'g{i}' for i in range(12)]
    expression_rows: list[np.ndarray] = []

    round1_effects = {
        'control': np.zeros(len(gene_names)),
        'A': np.array([2.0, 1.5, 1.0, 0.8] + [0.0] * 8),
        'B': np.array([0.8, 0.5, 0.2, 0.1, 1.5, 1.2] + [0.0] * 6),
    }
    round2_effects = {
        'control': np.zeros(len(gene_names)),
        'A': np.array([1.2, 1.0, 0.8, 0.4] + [0.0] * 8),
        'B': np.array([0.2, 0.2, 0.1, 0.1, 1.8, 1.5, 1.2] + [0.0] * 5),
    }

    for round1_treatment in ['control', 'A', 'B']:
        for i in range(n_per_group):
            base = rng.poisson(2.0, size=len(gene_names)).astype(float)
            line = f'line{(i % 2) + 1}'
            batch = f'batch{(i % 2) + 1}'
            replicate = f'r{(i % 2) + 1}'

            rows.append(
                {
                    'iPSC_line': line,
                    'batch': batch,
                    'replicate': replicate,
                    'round': 0,
                    'time_point': 'intermediate',
                    'round1_treatment': round1_treatment,
                    'round1_dose': 0.0 if round1_treatment == 'control' else (1.0 if round1_treatment == 'A' else 1.5),
                    'round2_treatment': 'control',
                    'round2_dose': 0.0,
                    'target_marker_score': 0.15 + 0.08 * (round1_treatment == 'A') + 0.03 * (round1_treatment == 'B'),
                    'stress_score': 0.04 + 0.05 * (round1_treatment == 'B'),
                    'off_target_score': 0.03 + 0.02 * (round1_treatment == 'B'),
                    'treatment_sequence': f'{round1_treatment}->control',
                }
            )
            expression_rows.append(base)

            post_round1 = base + round1_effects[round1_treatment] + rng.normal(0.0, 0.1, size=len(gene_names))
            rows.append(
                {
                    'iPSC_line': line,
                    'batch': batch,
                    'replicate': replicate,
                    'round': 1,
                    'time_point': 'post_round1',
                    'round1_treatment': round1_treatment,
                    'round1_dose': 0.0 if round1_treatment == 'control' else (1.0 if round1_treatment == 'A' else 1.5),
                    'round2_treatment': 'control',
                    'round2_dose': 0.0,
                    'target_marker_score': 0.35 + 0.18 * (round1_treatment == 'A') + 0.05 * (round1_treatment == 'B'),
                    'stress_score': 0.06 + 0.05 * (round1_treatment == 'B'),
                    'off_target_score': 0.03 + 0.02 * (round1_treatment == 'B'),
                    'treatment_sequence': f'{round1_treatment}->control',
                }
            )
            expression_rows.append(np.clip(post_round1, a_min=0.0, a_max=None))

            for round2_treatment in ['A', 'B']:
                post_round2 = post_round1 + round2_effects[round2_treatment] + rng.normal(0.0, 0.1, size=len(gene_names))
                target_bonus = 0.22 if (round1_treatment, round2_treatment) == ('A', 'B') else 0.12 if round2_treatment == 'A' else 0.04
                stress_penalty = 0.03 if round2_treatment == 'A' else 0.08
                off_target_penalty = 0.02 if round2_treatment == 'A' else 0.05
                rows.append(
                    {
                        'iPSC_line': line,
                        'batch': batch,
                        'replicate': replicate,
                        'round': 2,
                        'time_point': 'post_round2',
                        'round1_treatment': round1_treatment,
                        'round1_dose': 0.0 if round1_treatment == 'control' else (1.0 if round1_treatment == 'A' else 1.5),
                        'round2_treatment': round2_treatment,
                        'round2_dose': 0.5 if round2_treatment == 'A' else 1.0,
                        'target_marker_score': 0.45 + target_bonus,
                        'stress_score': 0.07 + stress_penalty + 0.02 * (round1_treatment == 'B'),
                        'off_target_score': 0.03 + off_target_penalty + 0.01 * (round1_treatment == 'B'),
                        'treatment_sequence': f'{round1_treatment}->{round2_treatment}',
                    }
                )
                expression_rows.append(np.clip(post_round2, a_min=0.0, a_max=None))

    obs = pd.DataFrame(rows)
    x = np.vstack(expression_rows).astype(np.float32)
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers['raw_counts'] = np.rint(x).astype(np.float32)
    adata.layers['log_normalized'] = np.log1p(np.clip(x, a_min=0.0, a_max=None)).astype(np.float32)
    return adata



def _treatment_library() -> list[dict[str, object]]:
    return [
        {'treatment_name': 'control', 'active': True, 'pathway': 'baseline'},
        {'treatment_name': 'A', 'active': True, 'pathway': 'wnt'},
        {'treatment_name': 'B', 'active': True, 'pathway': 'tgfb'},
    ]



def test_full_cpa_like_inverse_recommendation_e2e_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.full_cpa_like_adapter import FullCPALikeForwardAdapter
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig

    adata = _make_e2e_mock_adata()
    adapter = FullCPALikeForwardAdapter().fit(
        adata,
        FullCPALikeConfig(
            input_layer='raw_counts',
            covariate_keys=('iPSC_line', 'batch', 'round', 'time_point', 'replicate'),
            n_latent=6,
            n_hidden=16,
            n_layers=2,
            dropout=0.0,
            max_epochs=2,
            batch_size=8,
            device='cpu',
        ),
    )

    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    target_like = adata[
        (adata.obs['time_point'].astype(str) == 'post_round2')
        & (adata.obs['round1_treatment'].astype(str) == 'A')
        & (adata.obs['round2_treatment'].astype(str) == 'B')
    ].copy()

    recommendations = recommend_inverse_treatments(
        current_state=current,
        target_state=target_like,
        treatment_library=_treatment_library(),
        forward_model=adapter,
        top_n=100,
    )

    assert not recommendations.empty
    required_columns = {
        'recommendation_type',
        'round1_treatment',
        'round2_treatment',
        'treatment_sequence',
        'predicted_to_target_distance',
        'recommendation_score',
    }
    assert required_columns.issubset(recommendations.columns)
    assert recommendations['predicted_to_target_distance'].notna().all()
    assert recommendations['recommendation_score'].notna().all()
    assert recommendations['treatment_sequence'].astype(str).str.contains('->').any()

    row_ab = recommendations.loc[recommendations['treatment_sequence'] == 'A->B']
    row_ba = recommendations.loc[recommendations['treatment_sequence'] == 'B->A']
    assert not row_ab.empty
    assert not row_ba.empty
    assert float(row_ab.iloc[0]['recommendation_score']) != float(row_ba.iloc[0]['recommendation_score'])

    output_dir = tmp_path / 'inverse_e2e_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'full_cpa_like_e2e_model.pt'
    adapter.save(model_path)
    recommendations_path = output_dir / 'recommendations.csv'
    summary_path = output_dir / 'recommendation_summary.json'
    recommendations.to_csv(recommendations_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                'n_recommendations': int(len(recommendations)),
                'top_sequence': str(recommendations.iloc[0]['treatment_sequence']),
                'top_score': float(recommendations.iloc[0]['recommendation_score']),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    assert model_path.exists()
    assert recommendations_path.exists()
    assert summary_path.exists()

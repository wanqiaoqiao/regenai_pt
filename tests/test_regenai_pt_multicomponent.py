from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
from ipsc_digital_twin.models.regenai_pt_data import (
    build_regenai_pt_dataloaders,
    prepare_round_specific_adata,
)


def _make_multicomponent_adata(n_cells: int = 24) -> ad.AnnData:
    rows = []
    x = np.random.default_rng(0).poisson(2.0, size=(n_cells, 10)).astype(float)
    for idx in range(n_cells):
        cycle = idx % 3
        if cycle == 0:
            time_point = 'intermediate'
            round_value = 0
            round1_components = ['control']
            round1_doses = [0.0]
            round2_components = ['control']
            round2_doses = [0.0]
        elif cycle == 1:
            time_point = 'post_round1'
            round_value = 1
            round1_components = ['A', 'B'] if idx % 2 == 0 else ['A']
            round1_doses = [1.0, 2.0] if len(round1_components) == 2 else [1.0]
            round2_components = ['control']
            round2_doses = [0.0]
        else:
            time_point = 'post_round2'
            round_value = 2
            round1_components = ['A', 'B']
            round1_doses = [1.0, 2.0]
            round2_components = ['C', 'D'] if idx % 2 == 0 else ['C']
            round2_doses = [1.5, 0.5] if len(round2_components) == 2 else [1.5]
        rows.append(
            {
                'treatment': '+'.join(round1_components if round_value < 2 else round2_components),
                'dose': float(sum(round1_doses if round_value < 2 else round2_doses)),
                'round1_treatment': '+'.join(round1_components),
                'round1_dose': float(sum(round1_doses)),
                'round2_treatment': '+'.join(round2_components),
                'round2_dose': float(sum(round2_doses)),
                'round1_components': round1_components,
                'round1_doses': round1_doses,
                'round1_dose_units': ['uM'] * len(round1_components),
                'round2_components': round2_components,
                'round2_doses': round2_doses,
                'round2_dose_units': ['uM'] * len(round2_components),
                'treatment_components': round1_components if round_value < 2 else round2_components,
                'treatment_component_doses': round1_doses if round_value < 2 else round2_doses,
                'treatment_component_dose_units': ['uM'] * len(round1_components if round_value < 2 else round2_components),
                'iPSC_line': 'line1',
                'batch': 'batch1',
                'round': round_value,
                'time_point': time_point,
                'replicate': 'r1',
                'sequencing_run': 'run1',
            }
        )
    obs = pd.DataFrame(rows)
    var = pd.DataFrame(index=[f'g{i}' for i in range(x.shape[1])])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers['raw_counts'] = x.copy()
    return adata


def test_round_specific_preparation_keeps_multicomponent_fields() -> None:
    adata = _make_multicomponent_adata()
    config = RegenAIPTConfig(input_layer='raw_counts')
    round2 = prepare_round_specific_adata(adata, round_number=2, config=config)
    assert 'treatment_components' in round2.obs.columns
    assert 'treatment_component_doses' in round2.obs.columns
    assert 'round1_treatment' in round2.obs.columns
    assert isinstance(round2.obs['treatment_components'].iloc[0], list)


def test_multicomponent_dataloader_outputs_component_tensors_if_torch_available() -> None:
    torch = pytest.importorskip('torch')
    adata = _make_multicomponent_adata()
    config = RegenAIPTConfig(input_layer='raw_counts', max_components=3, batch_size=4)
    bundle = build_regenai_pt_dataloaders(adata, config)
    batch = next(iter(bundle.train_loader))
    assert isinstance(batch['component_ids'], torch.Tensor)
    assert isinstance(batch['component_doses'], torch.Tensor)
    assert isinstance(batch['component_mask'], torch.Tensor)
    assert batch['component_ids'].shape[1] == 3
    assert batch['component_mask'].shape == batch['component_doses'].shape
    assert float(batch['component_mask'].max().item()) == 1.0


def test_model_multicomponent_effect_differs_from_single_component_if_torch_available() -> None:
    torch = pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_model import RegenAIPTNet

    model = RegenAIPTNet(input_dim=6, n_treatments=5, n_latent=4, n_hidden=8, n_layers=2, dropout=0.0)
    x = torch.randn(2, 6)
    single_ids = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    single_doses = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    single_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    combo_ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    combo_doses = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    combo_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    pred_single = model.predict(x=x, component_ids=single_ids, component_doses=single_doses, component_mask=single_mask)
    pred_combo = model.predict(x=x, component_ids=combo_ids, component_doses=combo_doses, component_mask=combo_mask)
    assert not torch.allclose(pred_single, pred_combo)


def test_sequence_a_plus_b_then_c_plus_d_supported_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip('torch')
    from ipsc_digital_twin.models.regenai_pt_adapter import RegenAIPTForwardAdapter

    adata = _make_multicomponent_adata()
    config = RegenAIPTConfig(input_layer='raw_counts', max_epochs=1, batch_size=8, max_components=3)
    adapter = RegenAIPTForwardAdapter().fit(adata, config)
    current = adata[adata.obs['time_point'].astype(str) == 'intermediate'].copy()
    result = adapter.simulate_sequence(
        current,
        {
            'round1_components': ['A', 'B'],
            'round1_dose': [1.0, 2.0],
            'round2_components': ['C', 'D'],
            'round2_dose': [1.5, 0.5],
        },
    )
    assert 'predicted_state' in result
    assert result['x_hat'].shape[1] == adata.n_vars

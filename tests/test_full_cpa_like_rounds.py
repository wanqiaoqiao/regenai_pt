from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
from ipsc_digital_twin.models.full_cpa_like_data import prepare_round_specific_adata


def _make_round_adata(n_per_group: int = 6) -> ad.AnnData:
    rows = []
    for treatment in ["A", "B"]:
        for i in range(n_per_group):
            rows.append(
                {
                    "time_point": "intermediate",
                    "round": 0,
                    "round1_treatment": treatment,
                    "round1_dose": 1.0 if treatment == "A" else 2.0,
                    "round2_treatment": "control",
                    "round2_dose": 0.0,
                    "treatment_sequence": f"{treatment}->control",
                    "iPSC_line": f"line{(i % 2) + 1}",
                    "batch": "batch1",
                    "replicate": f"r{(i % 3) + 1}",
                    "sequencing_run": f"run{(i % 2) + 1}",
                }
            )
            for round2_treatment in ["A", "B"]:
                rows.append(
                    {
                        "time_point": "post_round1",
                        "round": 1,
                        "round1_treatment": treatment,
                        "round1_dose": 1.0 if treatment == "A" else 2.0,
                        "round2_treatment": round2_treatment,
                        "round2_dose": 0.5 if round2_treatment == "A" else 1.5,
                        "treatment_sequence": f"{treatment}->{round2_treatment}",
                        "iPSC_line": f"line{(i % 2) + 1}",
                        "batch": "batch1",
                        "replicate": f"r{(i % 3) + 1}",
                        "sequencing_run": f"run{(i % 2) + 1}",
                    }
                )
                rows.append(
                    {
                        "time_point": "post_round2",
                        "round": 2,
                        "round1_treatment": treatment,
                        "round1_dose": 1.0 if treatment == "A" else 2.0,
                        "round2_treatment": round2_treatment,
                        "round2_dose": 0.5 if round2_treatment == "A" else 1.5,
                        "treatment_sequence": f"{treatment}->{round2_treatment}",
                        "iPSC_line": f"line{(i % 2) + 1}",
                        "batch": "batch1",
                        "replicate": f"r{(i % 3) + 1}",
                        "sequencing_run": f"run{(i % 2) + 1}",
                    }
                )
    obs = pd.DataFrame(rows)
    x = np.random.poisson(2.0, size=(len(obs), 10)).astype(float)
    var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["raw_counts"] = x.copy()
    adata.layers["log_normalized"] = np.log1p(x)
    return adata


def test_prepare_round1_adata_works() -> None:
    adata = _make_round_adata()
    round1 = prepare_round_specific_adata(adata, round_number=1)
    assert set(round1.obs["time_point"].astype(str)) == {"intermediate", "post_round1"}
    assert set(round1.obs["treatment_role"].astype(str)) == {"round1_treatment"}
    assert "treatment" in round1.obs.columns
    assert "dose" in round1.obs.columns
    assert set(round1.obs["treatment"].astype(str)) == {"control", "A", "B"}


def test_prepare_round2_adata_works_and_keeps_round1_context() -> None:
    adata = _make_round_adata()
    round2 = prepare_round_specific_adata(adata, round_number=2)
    assert set(round2.obs["time_point"].astype(str)) == {"post_round1", "post_round2"}
    assert set(round2.obs["treatment_role"].astype(str)) == {"round2_treatment"}
    assert "round1_treatment" in round2.obs.columns
    assert set(round2.obs["treatment"].astype(str)) == {"control", "A", "B"}


def test_round2_dataloaders_include_round1_treatment_as_covariate_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_data import build_full_cpa_like_dataloaders

    adata = _make_round_adata()
    round2 = prepare_round_specific_adata(adata, round_number=2)
    config = FullCPALikeConfig(
        input_layer="raw_counts",
        covariate_keys=("replicate", "sequencing_run"),
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )
    bundle = build_full_cpa_like_dataloaders(round2, config)
    assert "round1_treatment" in bundle.mappings.covariate_to_id


def test_simulate_two_round_sequence_returns_prediction_and_order_matters_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_trainer import FullCPALikeTrainer
    from ipsc_digital_twin.models.simulation import simulate_two_round_sequence

    adata = _make_round_adata()
    round1 = prepare_round_specific_adata(adata, round_number=1)
    round2 = prepare_round_specific_adata(adata, round_number=2)

    config1 = FullCPALikeConfig(
        input_layer="raw_counts",
        covariate_keys=("replicate", "sequencing_run"),
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )
    config2 = FullCPALikeConfig(
        input_layer="raw_counts",
        covariate_keys=("replicate", "sequencing_run"),
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )

    trainer1 = FullCPALikeTrainer(config1).fit(round1)
    trainer2 = FullCPALikeTrainer(config2).fit(round2)

    current = adata[adata.obs["time_point"].astype(str) == "intermediate"].copy()
    seq_ab = simulate_two_round_sequence({"round1": trainer1, "round2": trainer2}, current, "A", "B")
    seq_ba = simulate_two_round_sequence({"round1": trainer1, "round2": trainer2}, current, "B", "A")

    assert seq_ab["x_hat"].shape == (current.n_obs, current.n_vars)
    assert seq_ab["predicted_adata"].X.shape == (current.n_obs, current.n_vars)
    assert not np.allclose(seq_ab["x_hat"], seq_ba["x_hat"])

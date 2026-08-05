from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from ipsc_digital_twin.models.regenai_pt_config import (
    RegenAIPTConfig,
    validate_regenai_pt_adata,
)


def _make_valid_adata(n_per_treatment: int = 12) -> ad.AnnData:
    treatments = ["control", "A", "B"]
    rows = []
    for treatment in treatments:
        for i in range(n_per_treatment):
            rows.append(
                {
                    "treatment": treatment,
                    "dose": 0.0 if treatment == "control" else 1.0,
                    "batch": "batch1",
                    "iPSC_line": "line1" if i % 2 == 0 else "line2",
                    "round": 1 if treatment != "B" else 2,
                    "time_point": "intermediate" if treatment == "control" else "post_round1",
                    "replicate": f"r{(i % 3) + 1}",
                }
            )
    obs = pd.DataFrame(rows)
    x = np.random.poisson(2.0, size=(len(obs), 16)).astype(float)
    var = pd.DataFrame(index=[f"g{i}" for i in range(16)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["raw_counts"] = x.copy()
    adata.layers["log_normalized"] = np.log1p(x)
    return adata


def test_regenai_pt_config_defaults_work() -> None:
    config = RegenAIPTConfig()
    assert config.model_type == "regenai_pt"
    assert config.input_layer == "X"
    assert config.n_latent > 0
    assert config.device == "auto"
    assert config.warmup_epochs == 20
    assert config.ramp_epochs == 20
    assert config.max_adversarial_weight == 0.05
    assert config.gradient_clip_norm == 5.0


def test_regenai_pt_config_rejects_nonpositive_gradient_clip_norm() -> None:
    for value in (0.0, -1.0):
        try:
            RegenAIPTConfig(gradient_clip_norm=value)
        except ValueError as exc:
            assert "gradient_clip_norm must be positive" in str(exc)
        else:
            raise AssertionError("Expected invalid gradient_clip_norm to fail")


def test_validate_regenai_pt_adata_missing_treatment_key_fails_clearly() -> None:
    adata = _make_valid_adata()
    config = RegenAIPTConfig(treatment_key="perturbation")

    report = validate_regenai_pt_adata(adata, config)

    assert not report.is_valid
    assert any("Required adata.obs column missing: 'perturbation'" in error for error in report.errors)


def test_validate_regenai_pt_adata_missing_control_treatment_fails_clearly() -> None:
    adata = _make_valid_adata()
    adata.obs["treatment"] = adata.obs["treatment"].replace({"control": "vehicle"})
    config = RegenAIPTConfig(control_treatment="control")

    report = validate_regenai_pt_adata(adata, config)

    assert not report.is_valid
    assert any("Control treatment 'control'" in error for error in report.errors)


def test_validate_regenai_pt_adata_valid_mock_passes() -> None:
    adata = _make_valid_adata()
    config = RegenAIPTConfig(covariate_keys=("replicate",), input_layer="raw_counts")

    report = validate_regenai_pt_adata(adata, config)

    assert report.is_valid
    assert report.n_cells == adata.n_obs
    assert report.n_treatments == 3
    assert report.errors == []


def test_validate_regenai_pt_adata_warns_for_small_treatments() -> None:
    adata = _make_valid_adata(n_per_treatment=3)
    config = RegenAIPTConfig()

    report = validate_regenai_pt_adata(adata, config)

    assert report.is_valid
    assert any("low cell counts" in warning for warning in report.warnings)

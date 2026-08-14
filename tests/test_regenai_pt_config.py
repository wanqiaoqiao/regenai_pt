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
    assert config.lr_scheduler_factor == 0.5
    assert config.lr_scheduler_patience == 5
    assert config.early_stopping_patience == 15
    assert config.gradient_clip_norm == 5.0
    assert config.n_de_genes == 100
    assert config.cell_type_key == "time_point"
    assert config.de_loss_weight == 0.0
    assert config.delta_loss_weight == 0.0
    assert config.delta_context_keys == ("iPSC_line", "round1_treatment")
    assert config.round1_component_durations_key == "round1_durations_hours"
    assert config.round2_component_durations_key == "round2_durations_hours"
    assert config.component_duration_key == "treatment_component_durations_hours"
    assert config.duration_regularization_weight == 0.1


def test_regenai_pt_config_rejects_invalid_lr_scheduler_settings() -> None:
    for factor in (0.0, 1.0, -0.5):
        try:
            RegenAIPTConfig(lr_scheduler_factor=factor)
        except ValueError as exc:
            assert "lr_scheduler_factor must be in the range (0, 1)" in str(exc)
        else:
            raise AssertionError("Expected invalid lr_scheduler_factor to fail")

    try:
        RegenAIPTConfig(lr_scheduler_patience=-1)
    except ValueError as exc:
        assert "lr_scheduler_patience must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected invalid lr_scheduler_patience to fail")


def test_regenai_pt_config_rejects_nonpositive_gradient_clip_norm() -> None:
    for value in (0.0, -1.0):
        try:
            RegenAIPTConfig(gradient_clip_norm=value)
        except ValueError as exc:
            assert "gradient_clip_norm must be positive" in str(exc)
        else:
            raise AssertionError("Expected invalid gradient_clip_norm to fail")


def test_regenai_pt_config_rejects_nonpositive_early_stopping_patience() -> None:
    for value in (0, -1):
        try:
            RegenAIPTConfig(early_stopping_patience=value)
        except ValueError as exc:
            assert "early_stopping_patience must be positive" in str(exc)
        else:
            raise AssertionError("Expected invalid early_stopping_patience to fail")


def test_regenai_pt_config_rejects_invalid_extended_metric_settings() -> None:
    with np.testing.assert_raises_regex(ValueError, "n_de_genes must be positive"):
        RegenAIPTConfig(n_de_genes=0)
    with np.testing.assert_raises_regex(ValueError, "cell_type_key must be a non-empty string"):
        RegenAIPTConfig(cell_type_key="")
    with np.testing.assert_raises_regex(ValueError, "de_loss_weight must be non-negative"):
        RegenAIPTConfig(de_loss_weight=-1.0)
    with np.testing.assert_raises_regex(ValueError, "delta_loss_weight must be non-negative"):
        RegenAIPTConfig(delta_loss_weight=-1.0)
    with np.testing.assert_raises_regex(ValueError, "duration_regularization_weight must be non-negative"):
        RegenAIPTConfig(duration_regularization_weight=-1.0)


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

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _make_mock_adata(n_cells: int = 36) -> ad.AnnData:
    treatments = ["control", "A", "B"]
    rows = []
    for idx in range(n_cells):
        treatment = treatments[idx % len(treatments)]
        rows.append(
            {
                "treatment": treatment,
                "dose": 0.0 if treatment == "control" else float((idx % 3) + 1),
                "iPSC_line": f"line{(idx % 2) + 1}",
                "batch": f"batch{(idx % 2) + 1}",
                "round": (idx % 2) + 1,
                "time_point": "intermediate" if treatment == "control" else "post_round1",
                "replicate": f"r{(idx % 3) + 1}",
                "sequencing_run": f"run{(idx % 2) + 1}",
            }
        )
    obs = pd.DataFrame(rows)
    x = np.random.poisson(2.0, size=(n_cells, 14)).astype(float)
    var = pd.DataFrame(index=[f"g{i}" for i in range(14)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["raw_counts"] = x.copy()
    adata.layers["log_normalized"] = np.log1p(x)
    return adata


def test_full_cpa_like_trainer_fit_encode_predict_and_save_load_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_trainer import FullCPALikeTrainer

    adata = _make_mock_adata()
    config = FullCPALikeConfig(
        input_layer="raw_counts",
        covariate_keys=("replicate", "sequencing_run"),
        n_latent=5,
        n_hidden=10,
        n_layers=2,
        dropout=0.0,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu",
    )
    trainer = FullCPALikeTrainer(config)
    trainer.fit(adata)

    assert trainer.model is not None
    assert trainer.mappings is not None
    assert trainer.epochs_trained >= 1
    loss_components = (
        "reconstruction_loss",
        "treatment_adv_loss",
        "covariate_adv_loss",
        "embedding_l2_loss",
        "dose_regularization_loss",
        "total_loss",
    )
    for split in ("train", "val"):
        for component in loss_components:
            values = trainer.history[f"{split}_{component}"]
            assert len(values) == trainer.epochs_trained
            assert np.isfinite(values).all()
    assert len(trainer.history["adversarial_weight"]) == trainer.epochs_trained
    assert trainer.history["adversarial_weight"] == [0.0] * trainer.epochs_trained

    z = trainer.encode_adata(adata)
    assert z.shape == (adata.n_obs, config.n_latent)

    preds = trainer.predict_adata(adata, treatment="A", dose=1.5)
    assert preds["x_hat"].shape == (adata.n_obs, adata.n_vars)
    assert preds["z_basal"].shape == (adata.n_obs, config.n_latent)
    assert preds["z_total"].shape == (adata.n_obs, config.n_latent)
    assert preds["dose_scale"].shape == (adata.n_obs, 1)

    save_path = tmp_path / "trainer.pt"
    trainer.save(save_path)
    assert save_path.exists()

    loaded = FullCPALikeTrainer.load(save_path)
    assert loaded.model is not None
    assert loaded.mappings is not None
    assert loaded.history.keys() == trainer.history.keys()

    preds_loaded = loaded.predict_adata(adata, treatment="A", dose=1.5)
    assert preds_loaded["x_hat"].shape == preds["x_hat"].shape


def test_adversarial_warmup_and_ramp_schedule_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_trainer import FullCPALikeTrainer

    trainer = FullCPALikeTrainer(
        FullCPALikeConfig(
            warmup_epochs=2,
            ramp_epochs=2,
            max_adversarial_weight=0.05,
            device="cpu",
        )
    )

    assert trainer.adversarial_weight_for_epoch(0) == 0.0
    assert trainer.adversarial_weight_for_epoch(1) == 0.0
    assert trainer.adversarial_weight_for_epoch(2) == pytest.approx(0.025)
    assert trainer.adversarial_weight_for_epoch(3) == pytest.approx(0.05)
    assert trainer.adversarial_weight_for_epoch(4) == pytest.approx(0.05)


def test_gradient_clipping_is_applied_during_training_if_torch_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_trainer import FullCPALikeTrainer

    calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters: object, max_norm: float, *args: object, **kwargs: object) -> object:
        calls.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    config = FullCPALikeConfig(
        input_layer="raw_counts",
        gradient_clip_norm=2.5,
        max_epochs=1,
        batch_size=8,
        n_hidden=8,
        n_latent=4,
        device="cpu",
    )

    FullCPALikeTrainer(config).fit(_make_mock_adata(n_cells=18))

    assert calls
    assert set(calls) == {2.5}


def test_full_cpa_like_trainer_predict_uses_string_and_scalar_covariates_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_trainer import FullCPALikeTrainer

    adata = _make_mock_adata(n_cells=18)
    config = FullCPALikeConfig(
        input_layer="raw_counts",
        covariate_keys=("replicate",),
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
        max_epochs=1,
        batch_size=6,
        device="cpu",
    )
    trainer = FullCPALikeTrainer(config).fit(adata)
    preds = trainer.predict_adata(
        adata,
        treatment="B",
        dose=2.0,
        covariates={"batch": "batch1", "iPSC_line": "line1", "replicate": "r1", "round": 1, "time_point": "post_round1"},
    )
    assert preds["x_hat"].shape == (adata.n_obs, adata.n_vars)

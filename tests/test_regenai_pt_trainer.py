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


def test_regenai_pt_trainer_fit_encode_predict_and_save_load_if_torch_available(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    adata = _make_mock_adata()
    config = RegenAIPTConfig(
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
    trainer = RegenAIPTTrainer(config)
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
    assert len(trainer.history["learning_rate"]) == trainer.epochs_trained
    assert np.isfinite(trainer.history["learning_rate"]).all()

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

    loaded = RegenAIPTTrainer.load(save_path)
    assert loaded.model is not None
    assert loaded.mappings is not None
    assert loaded.history.keys() == trainer.history.keys()

    preds_loaded = loaded.predict_adata(adata, treatment="A", dose=1.5)
    assert preds_loaded["x_hat"].shape == preds["x_hat"].shape


def test_adversarial_warmup_and_ramp_schedule_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    trainer = RegenAIPTTrainer(
        RegenAIPTConfig(
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
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters: object, max_norm: float, *args: object, **kwargs: object) -> object:
        calls.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    config = RegenAIPTConfig(
        input_layer="raw_counts",
        gradient_clip_norm=2.5,
        max_epochs=1,
        batch_size=8,
        n_hidden=8,
        n_latent=4,
        device="cpu",
    )

    RegenAIPTTrainer(config).fit(_make_mock_adata(n_cells=18))

    assert calls
    assert set(calls) == {2.5}


def test_reduce_lr_on_plateau_uses_configured_defaults_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    config = RegenAIPTConfig(learning_rate=0.01, device="cpu")
    trainer = RegenAIPTTrainer(config)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = trainer._initialize_lr_scheduler(optimizer)

    # The first call establishes the best value; six bad epochs exceed patience=5.
    for _ in range(7):
        scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_scheduler_monitors_validation_reconstruction_loss_if_torch_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    observed_metrics: list[float] = []
    original_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def recording_step(scheduler: object, metrics: float, *args: object, **kwargs: object) -> object:
        observed_metrics.append(float(metrics))
        return original_step(scheduler, metrics, *args, **kwargs)

    monkeypatch.setattr(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        "step",
        recording_step,
    )
    trainer = RegenAIPTTrainer(
        RegenAIPTConfig(
            input_layer="raw_counts",
            max_epochs=2,
            batch_size=8,
            n_hidden=8,
            n_latent=4,
            device="cpu",
        )
    ).fit(_make_mock_adata(n_cells=36))

    assert observed_metrics == pytest.approx(
        trainer.history["val_reconstruction_loss"]
    )


def test_regenai_pt_trainer_predict_uses_string_and_scalar_covariates_if_torch_available() -> None:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    adata = _make_mock_adata(n_cells=18)
    config = RegenAIPTConfig(
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
    trainer = RegenAIPTTrainer(config).fit(adata)
    preds = trainer.predict_adata(
        adata,
        treatment="B",
        dose=2.0,
        covariates={"batch": "batch1", "iPSC_line": "line1", "replicate": "r1", "round": 1, "time_point": "post_round1"},
    )
    assert preds["x_hat"].shape == (adata.n_obs, adata.n_vars)

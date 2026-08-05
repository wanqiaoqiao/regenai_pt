from __future__ import annotations

import pytest


def test_full_cpa_like_loss_components_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_losses import compute_full_cpa_like_loss
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    torch.manual_seed(0)
    model = FullCPALikeNet(
        input_dim=10,
        n_treatments=3,
        covariate_cardinalities={"batch": 2, "replicate": 3},
        n_latent=6,
        n_hidden=12,
        n_layers=2,
        dropout=0.0,
    )
    config = FullCPALikeConfig(
        reconstruction_loss="mse",
        adversarial_weight=1.0,
        covariate_adversarial_weight=0.5,
        perturbation_adversarial_weight=0.5,
        embedding_l2_weight=1e-4,
        dose_regularization_weight=0.1,
    )

    batch = {
        "x": torch.randn(4, 10),
        "treatment_id": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "dose_value": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        "covariate_ids": {
            "batch": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            "replicate": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        },
    }
    outputs = model(
        x=batch["x"],
        treatment_id=batch["treatment_id"],
        dose=batch["dose_value"],
        covariates=batch["covariate_ids"],
    )

    loss_dict = compute_full_cpa_like_loss(outputs=outputs, batch=batch, model=model, config=config)

    assert set(loss_dict) == {
        "total_loss",
        "reconstruction_loss",
        "treatment_adv_loss",
        "covariate_adv_loss",
        "embedding_l2_loss",
        "dose_regularization_loss",
    }
    assert isinstance(loss_dict["total_loss"], torch.Tensor)
    assert loss_dict["total_loss"].ndim == 0
    assert torch.isfinite(loss_dict["total_loss"])
    assert torch.isfinite(loss_dict["treatment_adv_loss"])
    assert torch.isfinite(loss_dict["covariate_adv_loss"])
    assert torch.isfinite(loss_dict["embedding_l2_loss"])
    assert torch.isfinite(loss_dict["dose_regularization_loss"])


def test_reconstruction_loss_decreases_when_prediction_matches_input_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_losses import compute_full_cpa_like_loss
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    model = FullCPALikeNet(
        input_dim=6,
        n_treatments=2,
        covariate_cardinalities={"batch": 2},
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
    )
    config = FullCPALikeConfig(reconstruction_loss="mse")

    x = torch.randn(3, 6)
    batch = {
        "x": x,
        "treatment_id": torch.tensor([0, 1, 1], dtype=torch.long),
        "dose_value": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        "covariate_ids": {"batch": torch.tensor([0, 1, 0], dtype=torch.long)},
    }
    outputs_bad = model(
        x=x,
        treatment_id=batch["treatment_id"],
        dose=batch["dose_value"],
        covariates=batch["covariate_ids"],
    )
    outputs_good = dict(outputs_bad)
    outputs_good["x_hat"] = x.clone()

    loss_bad = compute_full_cpa_like_loss(outputs=outputs_bad, batch=batch, model=model, config=config)
    loss_good = compute_full_cpa_like_loss(outputs=outputs_good, batch=batch, model=model, config=config)

    assert loss_good["reconstruction_loss"] <= loss_bad["reconstruction_loss"]
    assert torch.allclose(loss_good["reconstruction_loss"], torch.tensor(0.0), atol=1e-6)


def test_losses_remain_finite_without_covariates_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_losses import compute_full_cpa_like_loss
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    model = FullCPALikeNet(
        input_dim=5,
        n_treatments=2,
        covariate_cardinalities={},
        n_latent=3,
        n_hidden=6,
        n_layers=2,
        dropout=0.0,
    )
    config = FullCPALikeConfig(reconstruction_loss="mse")
    batch = {
        "x": torch.randn(2, 5),
        "treatment_id": torch.tensor([0, 1], dtype=torch.long),
        "dose_value": torch.tensor([0.0, 2.0], dtype=torch.float32),
        "covariate_ids": {},
    }
    outputs = model(x=batch["x"], treatment_id=batch["treatment_id"], dose=batch["dose_value"], covariates=None)
    loss_dict = compute_full_cpa_like_loss(outputs=outputs, batch=batch, model=model, config=config)

    assert torch.isfinite(loss_dict["covariate_adv_loss"])
    assert torch.isfinite(loss_dict["embedding_l2_loss"])
    assert torch.isfinite(loss_dict["dose_regularization_loss"])


def test_zero_active_adversarial_weight_excludes_adversarial_losses_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_config import FullCPALikeConfig
    from ipsc_digital_twin.models.full_cpa_like_losses import compute_full_cpa_like_loss
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    model = FullCPALikeNet(input_dim=5, n_treatments=2, n_latent=3, n_hidden=6)
    config = FullCPALikeConfig(
        embedding_l2_weight=0.0,
        dose_regularization_weight=0.0,
    )
    batch = {
        "x": torch.randn(3, 5),
        "treatment_id": torch.tensor([0, 1, 1]),
        "dose_value": torch.tensor([0.0, 1.0, 2.0]),
        "covariate_ids": {},
    }
    outputs = model(
        x=batch["x"],
        treatment_id=batch["treatment_id"],
        dose=batch["dose_value"],
    )
    losses = compute_full_cpa_like_loss(
        outputs=outputs,
        batch=batch,
        model=model,
        config=config,
        active_adversarial_weight=0.0,
    )

    assert torch.allclose(losses["total_loss"], losses["reconstruction_loss"])

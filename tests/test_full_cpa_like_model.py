from __future__ import annotations

import pytest


def test_full_cpa_like_model_forward_shapes_and_behavior_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    torch.manual_seed(0)
    model = FullCPALikeNet(
        input_dim=12,
        n_treatments=4,
        covariate_cardinalities={"batch": 3, "iPSC_line": 2},
        n_latent=8,
        n_hidden=16,
        n_layers=2,
        dropout=0.0,
    )

    x = torch.randn(5, 12)
    treatment_id = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
    dose = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32)
    covariates = {
        "batch": torch.tensor([0, 1, 2, 1, 0], dtype=torch.long),
        "iPSC_line": torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
    }

    outputs = model(x=x, treatment_id=treatment_id, dose=dose, covariates=covariates)

    assert outputs["x_hat"].shape == (5, 12)
    assert outputs["z_basal"].shape == (5, 8)
    assert outputs["z_total"].shape == (5, 8)
    assert outputs["perturbation_embedding"].shape == (5, 8)
    assert outputs["dose_scale"].shape == (5, 1)
    assert outputs["treatment_logits_adv"].shape == (5, 4)
    assert outputs["covariate_logits_adv"]["batch"].shape == (5, 3)
    assert outputs["covariate_logits_adv"]["iPSC_line"].shape == (5, 2)


def test_encode_returns_latent_and_embeddings_differ_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    torch.manual_seed(1)
    model = FullCPALikeNet(
        input_dim=10,
        n_treatments=3,
        covariate_cardinalities={"batch": 2},
        n_latent=6,
        n_hidden=12,
        n_layers=2,
        dropout=0.0,
    )

    x = torch.randn(4, 10)
    z = model.encode(x)
    assert z.shape == (4, 6)

    embeddings = model.get_treatment_embeddings()
    assert embeddings.shape == (3, 6)
    assert not torch.allclose(embeddings[0], embeddings[1])


def test_dose_changes_prediction_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    torch.manual_seed(2)
    model = FullCPALikeNet(
        input_dim=8,
        n_treatments=2,
        covariate_cardinalities={"batch": 2},
        n_latent=4,
        n_hidden=8,
        n_layers=2,
        dropout=0.0,
    )
    model.treatment_dose_a.weight.data.fill_(1.0)
    model.treatment_dose_b.weight.data.zero_()

    x = torch.randn(2, 8)
    treatment_id = torch.tensor([1, 1], dtype=torch.long)
    covariates = {"batch": torch.tensor([0, 0], dtype=torch.long)}

    pred_low = model.predict(x=x, treatment_id=treatment_id, dose=torch.tensor([0.0, 0.0]), covariates=covariates)
    pred_high = model.predict(x=x, treatment_id=treatment_id, dose=torch.tensor([5.0, 5.0]), covariates=covariates)

    assert pred_low.shape == pred_high.shape == (2, 8)
    assert not torch.allclose(pred_low, pred_high)


def test_decode_and_covariate_embeddings_if_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from ipsc_digital_twin.models.full_cpa_like_model import FullCPALikeNet

    model = FullCPALikeNet(
        input_dim=6,
        n_treatments=2,
        covariate_cardinalities={"batch": 2, "replicate": 3},
        n_latent=5,
        n_hidden=10,
        n_layers=2,
        dropout=0.0,
    )

    z_basal = torch.randn(3, 5)
    treatment_id = torch.tensor([0, 1, 1], dtype=torch.long)
    dose = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    covariates = {
        "batch": torch.tensor([0, 1, 0], dtype=torch.long),
        "replicate": torch.tensor([0, 1, 2], dtype=torch.long),
    }

    x_hat, z_total, perturbation_embedding, dose_scale = model.decode(
        z_basal=z_basal,
        treatment_id=treatment_id,
        dose=dose,
        covariates=covariates,
    )

    assert x_hat.shape == (3, 6)
    assert z_total.shape == (3, 5)
    assert perturbation_embedding.shape == (3, 5)
    assert dose_scale.shape == (3, 1)

    cov_embs = model.get_covariate_embeddings()
    assert set(cov_embs) == {"batch", "replicate"}
    assert cov_embs["batch"].shape == (2, 5)
    assert cov_embs["replicate"].shape == (3, 5)

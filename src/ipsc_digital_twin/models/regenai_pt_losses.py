from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised in torch-enabled environments
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for RegenAI-PT losses. Install torch to use the '--transition-model regenai_pt' backend."
    ) from exc

from .regenai_pt_config import RegenAIPTConfig


def _zero_tensor(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)



def _compute_reconstruction_loss(outputs: dict[str, Any], batch: dict[str, Any], config: RegenAIPTConfig) -> torch.Tensor:
    x = batch["x"].float()
    x_hat = outputs["x_hat"].float()
    if config.reconstruction_loss == "mse":
        return F.mse_loss(x_hat, x)
    if config.reconstruction_loss in {"nb", "zinb"}:
        raise NotImplementedError(
            f"reconstruction_loss='{config.reconstruction_loss}' is not implemented yet; use 'mse' for now"
        )
    raise ValueError(f"Unsupported reconstruction_loss: {config.reconstruction_loss}")



def _compute_treatment_adv_loss(outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    logits = outputs["treatment_logits_adv"]
    targets = batch["treatment_id"].long()
    return F.cross_entropy(logits, targets)



def _compute_covariate_adv_loss(outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    logits_by_covariate: dict[str, torch.Tensor] = outputs.get("covariate_logits_adv", {})
    covariate_targets: dict[str, torch.Tensor] = batch.get("covariate_ids", {})
    if not logits_by_covariate:
        ref = outputs["x_hat"]
        return _zero_tensor(ref)

    losses: list[torch.Tensor] = []
    for key, logits in logits_by_covariate.items():
        if key not in covariate_targets:
            continue
        losses.append(F.cross_entropy(logits, covariate_targets[key].long()))
    if not losses:
        return _zero_tensor(outputs["x_hat"])
    return torch.stack(losses).mean()



def _compute_embedding_l2_loss(model: Any, reference: torch.Tensor) -> torch.Tensor:
    penalties = [model.component_embedding.weight.pow(2).mean()]
    for embedding in model.covariate_embeddings.values():
        penalties.append(embedding.weight.pow(2).mean())
    return torch.stack(penalties).mean() if penalties else _zero_tensor(reference)



def _compute_dose_regularization_loss(model: Any, reference: torch.Tensor) -> torch.Tensor:
    penalties = [
        model.component_dose_a.weight.pow(2).mean(),
        model.component_dose_b.weight.pow(2).mean(),
    ]
    return torch.stack(penalties).mean() if penalties else _zero_tensor(reference)



def compute_regenai_pt_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    model: Any,
    config: RegenAIPTConfig,
) -> dict[str, torch.Tensor]:
    reconstruction_loss = _compute_reconstruction_loss(outputs=outputs, batch=batch, config=config)
    treatment_adv_loss = _compute_treatment_adv_loss(outputs=outputs, batch=batch)
    covariate_adv_loss = _compute_covariate_adv_loss(outputs=outputs, batch=batch)
    embedding_l2_loss = _compute_embedding_l2_loss(model=model, reference=reconstruction_loss)
    dose_regularization_loss = _compute_dose_regularization_loss(model=model, reference=reconstruction_loss)

    total_loss = reconstruction_loss
    total_loss = total_loss + (
        config.adversarial_weight * config.perturbation_adversarial_weight * treatment_adv_loss
    )
    total_loss = total_loss + (
        config.adversarial_weight * config.covariate_adversarial_weight * covariate_adv_loss
    )
    total_loss = total_loss + (config.embedding_l2_weight * embedding_l2_loss)
    total_loss = total_loss + (config.dose_regularization_weight * dose_regularization_loss)

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "treatment_adv_loss": treatment_adv_loss,
        "covariate_adv_loss": covariate_adv_loss,
        "embedding_l2_loss": embedding_l2_loss,
        "dose_regularization_loss": dose_regularization_loss,
    }


__all__ = ["compute_regenai_pt_loss"]

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised indirectly in torch-enabled environments
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for RegenAIPTNet. Install torch to use the '--transition-model regenai_pt' backend."
    ) from exc


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, n_layers - 1)):
            layers.extend([
                nn.Linear(in_dim, n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = n_hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RegenAIPTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_treatments: int,
        n_components: int | None = None,
        covariate_cardinalities: dict[str, int] | None = None,
        n_latent: int = 32,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        adversarial_lambda: float = 1.0,
        max_components: int = 4,
        use_component_interactions: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_treatments <= 0:
            raise ValueError("n_treatments must be positive")
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_latent <= 0:
            raise ValueError("n_latent must be positive")

        self.input_dim = int(input_dim)
        self.n_treatments = int(n_treatments)
        self.n_components = int(n_components or n_treatments)
        self.n_latent = int(n_latent)
        self.max_components = int(max_components)
        self.use_component_interactions = bool(use_component_interactions)
        self.covariate_cardinalities = dict(covariate_cardinalities or {})

        self.encoder = _MLP(input_dim=input_dim, output_dim=n_latent, n_hidden=n_hidden, n_layers=n_layers, dropout=dropout)
        self.component_embedding = nn.Embedding(self.n_components, n_latent)
        self.component_dose_a = nn.Embedding(self.n_components, 1)
        self.component_dose_b = nn.Embedding(self.n_components, 1)
        self.covariate_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(num_embeddings=size, embedding_dim=n_latent)
                for key, size in self.covariate_cardinalities.items()
            }
        )
        self.decoder = _MLP(input_dim=n_latent, output_dim=input_dim, n_hidden=n_hidden, n_layers=n_layers, dropout=dropout)
        self.interaction_mlp = _MLP(input_dim=n_latent * 2, output_dim=n_latent, n_hidden=n_hidden, n_layers=2, dropout=dropout)

        self.gradient_reversal = GradientReversalLayer(lambda_=adversarial_lambda)
        self.treatment_classifier = _MLP(
            input_dim=n_latent,
            output_dim=n_treatments,
            n_hidden=n_hidden,
            n_layers=max(1, n_layers - 1),
            dropout=dropout,
        )
        self.covariate_classifiers = nn.ModuleDict(
            {
                key: _MLP(
                    input_dim=n_latent,
                    output_dim=size,
                    n_hidden=n_hidden,
                    n_layers=max(1, n_layers - 1),
                    dropout=dropout,
                )
                for key, size in self.covariate_cardinalities.items()
            }
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.component_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.component_dose_a.weight)
        nn.init.zeros_(self.component_dose_b.weight)
        for key in self.covariate_embeddings:
            embedding = self.covariate_embeddings[key]
            assert isinstance(embedding, nn.Embedding)
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    @property
    def treatment_embedding(self) -> nn.Embedding:
        return self.component_embedding

    @property
    def treatment_dose_a(self) -> nn.Embedding:
        return self.component_dose_a

    @property
    def treatment_dose_b(self) -> nn.Embedding:
        return self.component_dose_b

    def _compute_covariate_effect(self, covariates: dict[str, torch.Tensor] | None, batch_size: int) -> torch.Tensor:
        effect = torch.zeros(batch_size, self.n_latent, device=self.component_embedding.weight.device)
        if not covariates:
            return effect
        for key, ids in covariates.items():
            if key not in self.covariate_embeddings:
                continue
            effect = effect + self.covariate_embeddings[key](ids)
        return effect

    def _compute_component_dose_scale(self, component_ids: torch.Tensor, component_doses: torch.Tensor) -> torch.Tensor:
        dose = component_doses.float()
        log_dose = torch.log1p(torch.clamp(dose, min=0.0)).unsqueeze(-1)
        a = self.component_dose_a(component_ids)
        b = self.component_dose_b(component_ids)
        return torch.sigmoid(a * log_dose + b)

    def _normalize_component_inputs(
        self,
        *,
        component_ids: torch.Tensor | None,
        component_doses: torch.Tensor | None,
        component_mask: torch.Tensor | None,
        treatment_id: torch.Tensor | None,
        dose: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if component_ids is None:
            if treatment_id is None:
                raise ValueError("component_ids or treatment_id is required")
            component_ids = treatment_id.view(-1, 1)
            component_doses = (dose if dose is not None else torch.ones_like(treatment_id, dtype=torch.float32)).view(-1, 1)
            component_mask = torch.ones_like(component_doses, dtype=torch.float32)
        else:
            if component_doses is None:
                component_doses = torch.ones(component_ids.shape, device=component_ids.device, dtype=torch.float32)
            if component_mask is None:
                component_mask = torch.ones(component_ids.shape, device=component_ids.device, dtype=torch.float32)
        return component_ids.long(), component_doses.float(), component_mask.float()

    def _compute_component_effect(
        self,
        component_ids: torch.Tensor,
        component_doses: torch.Tensor,
        component_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        component_embedding = self.component_embedding(component_ids)
        dose_scale = self._compute_component_dose_scale(component_ids=component_ids, component_doses=component_doses)
        masked_scale = dose_scale * component_mask.unsqueeze(-1)
        masked_effects = component_embedding * masked_scale
        perturbation_embedding = masked_effects.sum(dim=1)
        return perturbation_embedding, component_embedding, dose_scale, masked_effects

    def _compute_interaction_effect(self, masked_effects: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        if not self.use_component_interactions or masked_effects.shape[1] < 2:
            return torch.zeros(masked_effects.shape[0], self.n_latent, device=masked_effects.device)
        first = masked_effects[:, :-1, :]
        second = masked_effects[:, 1:, :]
        pair_mask = (component_mask[:, :-1] * component_mask[:, 1:]).unsqueeze(-1)
        if pair_mask.numel() == 0:
            return torch.zeros(masked_effects.shape[0], self.n_latent, device=masked_effects.device)
        pair_features = torch.cat([first, second], dim=-1)
        pair_effect = self.interaction_mlp(pair_features.view(-1, self.n_latent * 2)).view(masked_effects.shape[0], -1, self.n_latent)
        return (pair_effect * pair_mask).sum(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(
        self,
        z_basal: torch.Tensor,
        treatment_id: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
        covariates: dict[str, torch.Tensor] | None = None,
        component_ids: torch.Tensor | None = None,
        component_doses: torch.Tensor | None = None,
        component_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        component_ids, component_doses, component_mask = self._normalize_component_inputs(
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
            treatment_id=treatment_id,
            dose=dose,
        )
        perturbation_embedding, _, dose_scale, masked_effects = self._compute_component_effect(
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
        )
        interaction_effect = self._compute_interaction_effect(masked_effects, component_mask)
        covariate_effect = self._compute_covariate_effect(covariates, batch_size=z_basal.shape[0])
        z_total = z_basal + perturbation_embedding + interaction_effect + covariate_effect
        x_hat = self.decoder(z_total)
        active_count = component_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        aggregate_dose_scale = (dose_scale.squeeze(-1) * component_mask).sum(dim=1, keepdim=True) / active_count
        return x_hat, z_total, perturbation_embedding, aggregate_dose_scale

    def forward(
        self,
        x: torch.Tensor,
        treatment_id: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
        covariates: dict[str, torch.Tensor] | None = None,
        component_ids: torch.Tensor | None = None,
        component_doses: torch.Tensor | None = None,
        component_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        z_basal = self.encode(x)
        x_hat, z_total, perturbation_embedding, dose_scale = self.decode(
            z_basal=z_basal,
            treatment_id=treatment_id,
            dose=dose,
            covariates=covariates,
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
        )
        normalized_ids, normalized_doses, _ = self._normalize_component_inputs(
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
            treatment_id=treatment_id,
            dose=dose,
        )
        component_dose_scale = self._compute_component_dose_scale(
            component_ids=normalized_ids,
            component_doses=normalized_doses,
        ).squeeze(-1)
        z_adv = self.gradient_reversal(z_basal)
        treatment_logits_adv = self.treatment_classifier(z_adv)
        covariate_logits_adv = {key: classifier(z_adv) for key, classifier in self.covariate_classifiers.items()}
        return {
            "x_hat": x_hat,
            "z_basal": z_basal,
            "z_total": z_total,
            "perturbation_embedding": perturbation_embedding,
            "dose_scale": dose_scale,
            "component_dose_scale": component_dose_scale,
            "treatment_logits_adv": treatment_logits_adv,
            "covariate_logits_adv": covariate_logits_adv,
        }

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        treatment_id: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
        covariates: dict[str, torch.Tensor] | None = None,
        component_ids: torch.Tensor | None = None,
        component_doses: torch.Tensor | None = None,
        component_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.forward(
            x=x,
            treatment_id=treatment_id,
            dose=dose,
            covariates=covariates,
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
        )
        return outputs["x_hat"]

    def get_treatment_embeddings(self) -> torch.Tensor:
        return self.component_embedding.weight.detach().clone()

    def get_covariate_embeddings(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key in self.covariate_embeddings:
            embedding = self.covariate_embeddings[key]
            assert isinstance(embedding, nn.Embedding)
            result[key] = embedding.weight.detach().clone()
        return result

__all__ = [
    "GradientReversalFunction",
    "GradientReversalLayer",
    "RegenAIPTNet",
]

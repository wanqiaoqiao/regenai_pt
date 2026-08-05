from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np

from .regenai_pt_config import RegenAIPTConfig, validate_regenai_pt_adata
from .regenai_pt_data import RegenAIPTMappings, build_regenai_pt_dataloaders, require_torch

try:  # pragma: no cover - exercised in torch-enabled environments
    import torch
    from torch.optim import AdamW
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for RegenAIPTTrainer. Install torch to use the '--transition-model regenai_pt' backend."
    ) from exc

from .regenai_pt_losses import compute_regenai_pt_loss
from .regenai_pt_model import RegenAIPTNet

LOGGER = logging.getLogger(__name__)

LOSS_COMPONENTS = (
    "reconstruction_loss",
    "treatment_adv_loss",
    "covariate_adv_loss",
    "embedding_l2_loss",
    "dose_regularization_loss",
    "total_loss",
)


class RegenAIPTTrainer:
    def __init__(self, config: RegenAIPTConfig) -> None:
        require_torch()
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model: RegenAIPTNet | None = None
        self.mappings: RegenAIPTMappings | None = None
        self.history: dict[str, list[float]] = {
            f"{split}_{component}": []
            for split in ("train", "val")
            for component in LOSS_COMPONENTS
        }
        self.history["adversarial_weight"] = []
        self.best_val_reconstruction_loss: float | None = None
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.epochs_trained: int = 0
        self.early_stopping_patience: int = 5

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("config.device='cuda' requested but CUDA is not available")
            return torch.device("cuda")
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _covariate_cardinalities(self, mappings: RegenAIPTMappings) -> dict[str, int]:
        return {key: len(value) for key, value in mappings.covariate_to_id.items()}

    def _initialize_model(self, mappings: RegenAIPTMappings) -> RegenAIPTNet:
        model = RegenAIPTNet(
            input_dim=mappings.input_dim,
            n_treatments=len(mappings.treatment_to_id),
            n_components=len(mappings.component_to_id),
            covariate_cardinalities=self._covariate_cardinalities(mappings),
            n_latent=self.config.n_latent,
            n_hidden=self.config.n_hidden,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            # The scheduled loss coefficient applies adversarial strength once.
            adversarial_lambda=1.0,
            max_components=mappings.max_components,
            use_component_interactions=self.config.use_component_interactions,
        )
        return model.to(self.device)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "x": batch["x"].to(self.device),
            "treatment_id": batch["treatment_id"].to(self.device),
            "dose_value": batch["dose_value"].to(self.device),
            "component_ids": batch["component_ids"].to(self.device),
            "component_doses": batch["component_doses"].to(self.device),
            "component_mask": batch["component_mask"].to(self.device),
            "covariate_ids": {key: value.to(self.device) for key, value in batch.get("covariate_ids", {}).items()},
            "round_id": batch["round_id"].to(self.device),
            "time_id": batch["time_id"].to(self.device),
            "batch_id": batch["batch_id"].to(self.device),
            "sample_weight": batch["sample_weight"].to(self.device),
        }

    def adversarial_weight_for_epoch(self, epoch_index: int) -> float:
        """Return the scheduled adversarial coefficient for a zero-based epoch."""
        if epoch_index < 0:
            raise ValueError("epoch_index must be non-negative")
        if epoch_index < self.config.warmup_epochs:
            return 0.0
        if self.config.ramp_epochs == 0:
            return float(self.config.max_adversarial_weight)
        ramp_step = epoch_index - self.config.warmup_epochs + 1
        progress = min(max(ramp_step / self.config.ramp_epochs, 0.0), 1.0)
        return float(self.config.max_adversarial_weight * progress)

    def fit(self, adata: ad.AnnData) -> RegenAIPTTrainer:
        report = validate_regenai_pt_adata(adata, self.config)
        if not report.is_valid:
            raise ValueError("Invalid AnnData for RegenAI-PT training: " + "; ".join(report.errors))

        data_bundle = build_regenai_pt_dataloaders(adata=adata, config=self.config)
        self.mappings = data_bundle.mappings
        self.model = self._initialize_model(self.mappings)
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        best_val = float("inf")
        patience_counter = 0
        schedule_complete_epoch = max(
            self.config.warmup_epochs + self.config.ramp_epochs - 1,
            0,
        )
        for epoch in range(self.config.max_epochs):
            active_adversarial_weight = self.adversarial_weight_for_epoch(epoch)
            train_metrics = self.train_epoch(
                data_bundle.train_loader,
                optimizer,
                active_adversarial_weight=active_adversarial_weight,
            )
            val_metrics = self.validate_epoch(
                data_bundle.val_loader,
                active_adversarial_weight=active_adversarial_weight,
            )
            val_recon = val_metrics["reconstruction_loss"]
            if np.isnan(val_recon):
                val_recon = train_metrics["reconstruction_loss"]
            for component in LOSS_COMPONENTS:
                self.history[f"train_{component}"].append(train_metrics[component])
                self.history[f"val_{component}"].append(val_metrics[component])
            self.history["adversarial_weight"].append(active_adversarial_weight)
            LOGGER.info(
                "Epoch %d/%d | adversarial_weight=%.6f | train %s | val %s",
                epoch + 1,
                self.config.max_epochs,
                active_adversarial_weight,
                " ".join(f"{key}={train_metrics[key]:.6f}" for key in LOSS_COMPONENTS),
                " ".join(f"{key}={val_metrics[key]:.6f}" for key in LOSS_COMPONENTS),
            )
            self.epochs_trained = epoch + 1
            checkpoint_eligible = (
                epoch >= schedule_complete_epoch
                or epoch == self.config.max_epochs - 1
            )
            if checkpoint_eligible and val_recon < best_val:
                best_val = val_recon
                self.best_val_reconstruction_loss = val_recon
                self.best_state_dict = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience_counter = 0
            elif checkpoint_eligible:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
            else:
                # Do not stop or select a checkpoint before adversarial ramp-up
                # has completed; that would silently restore a warm-up model.
                patience_counter = 0
        if self.best_state_dict is not None and self.model is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self

    def train_epoch(
        self,
        train_loader: Any,
        optimizer: AdamW,
        active_adversarial_weight: float | None = None,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        self.model.train()
        accum = {key: 0.0 for key in LOSS_COMPONENTS}
        n_batches = 0
        for batch in train_loader:
            batch = self._move_batch_to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(
                x=batch["x"],
                treatment_id=batch["treatment_id"],
                dose=batch["dose_value"],
                covariates=batch["covariate_ids"],
                component_ids=batch["component_ids"],
                component_doses=batch["component_doses"],
                component_mask=batch["component_mask"],
            )
            loss_dict = compute_regenai_pt_loss(
                outputs=outputs,
                batch=batch,
                model=self.model,
                config=self.config,
                active_adversarial_weight=active_adversarial_weight,
            )
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm,
            )
            optimizer.step()
            for key in accum:
                accum[key] += float(loss_dict[key].detach().cpu().item())
            n_batches += 1
        if n_batches == 0:
            return {key: float("nan") for key in accum}
        return {key: value / n_batches for key, value in accum.items()}

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: Any,
        active_adversarial_weight: float | None = None,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model has not been initialized")
        self.model.eval()
        accum = {key: 0.0 for key in LOSS_COMPONENTS}
        n_batches = 0
        for batch in val_loader:
            batch = self._move_batch_to_device(batch)
            outputs = self.model(
                x=batch["x"],
                treatment_id=batch["treatment_id"],
                dose=batch["dose_value"],
                covariates=batch["covariate_ids"],
                component_ids=batch["component_ids"],
                component_doses=batch["component_doses"],
                component_mask=batch["component_mask"],
            )
            loss_dict = compute_regenai_pt_loss(
                outputs=outputs,
                batch=batch,
                model=self.model,
                config=self.config,
                active_adversarial_weight=active_adversarial_weight,
            )
            for key in accum:
                accum[key] += float(loss_dict[key].detach().cpu().item())
            n_batches += 1
        if n_batches == 0:
            return {key: float("nan") for key in accum}
        return {key: value / n_batches for key, value in accum.items()}

    def _ensure_fitted(self) -> None:
        if self.model is None or self.mappings is None:
            raise RuntimeError("Trainer has not been fitted or loaded")

    def _expression_from_adata(self, adata: ad.AnnData) -> np.ndarray:
        if self.config.input_layer == "X":
            matrix = adata.X
        elif self.config.input_layer in adata.layers:
            matrix = adata.layers[self.config.input_layer]
        else:
            # Sequential predictions are generated in X and do not necessarily
            # carry the source layer name into the synthetic AnnData.
            matrix = adata.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=np.float32)

    def _encode_covariates_for_prediction(self, n_obs: int, adata: ad.AnnData, covariates: dict[str, Any] | None) -> dict[str, torch.Tensor]:
        assert self.mappings is not None
        covariate_ids: dict[str, torch.Tensor] = {}
        covariates = covariates or {}
        for key, mapping in self.mappings.covariate_to_id.items():
            if key in covariates:
                raw = covariates[key]
                values = [str(raw)] * n_obs if np.isscalar(raw) else [str(value) for value in raw]
            elif key in adata.obs.columns:
                values = adata.obs[key].astype(str).tolist()
            else:
                first = next(iter(mapping.keys()))
                values = [str(first)] * n_obs

            fallback_id = next(iter(mapping.values()))
            encoded_values: list[int] = []
            for value in values:
                label = str(value)
                if label in mapping:
                    encoded_values.append(mapping[label])
                    continue
                known_component = next(
                    (part.strip() for part in label.split("+") if part.strip() in mapping),
                    None,
                )
                encoded_values.append(mapping[known_component] if known_component is not None else fallback_id)
            covariate_ids[key] = torch.as_tensor(encoded_values, dtype=torch.long, device=self.device)
        return covariate_ids

    def _normalize_component_spec(self, treatment: Any, dose: Any, n_obs: int) -> tuple[list[list[str]], list[list[float]]]:
        if isinstance(treatment, str):
            components = [[part.strip() for part in treatment.split('+') if part.strip()]] * n_obs
        elif isinstance(treatment, (list, tuple)) and treatment and all(isinstance(item, str) for item in treatment):
            components = [[str(item) for item in treatment]] * n_obs
        elif isinstance(treatment, (list, tuple)) and len(treatment) == n_obs and all(isinstance(item, (list, tuple)) for item in treatment):
            components = [[str(part) for part in item] for item in treatment]
        else:
            raise KeyError(f"Unknown treatment specification: {treatment}")

        if dose is None:
            doses = [[1.0] * len(row) for row in components]
        elif np.isscalar(dose):
            scalar_dose = float(cast(float | int | str, dose))
            doses = [[scalar_dose] * len(row) for row in components]
        elif isinstance(dose, (list, tuple)) and dose and all(np.isscalar(item) for item in dose):
            dose_list = [float(item) for item in dose]
            if len(dose_list) == len(components[0]):
                doses = [dose_list[:] for _ in range(n_obs)]
            else:
                doses = [[dose_list[idx]] * len(row) for idx, row in enumerate(components)]
        else:
            doses = [[float(value) for value in row] for row in dose]
        return components, doses

    def _encode_components_for_prediction(self, treatment: Any, dose: Any, n_obs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.mappings is not None
        component_rows, dose_rows = self._normalize_component_spec(treatment, dose, n_obs)
        component_ids = np.zeros((n_obs, self.config.max_components), dtype=np.int64)
        component_doses = np.zeros((n_obs, self.config.max_components), dtype=np.float32)
        component_mask = np.zeros((n_obs, self.config.max_components), dtype=np.float32)
        treatment_labels: list[str] = []
        for row_idx, (components, doses) in enumerate(zip(component_rows, dose_rows, strict=False)):
            treatment_labels.append('+'.join(components))
            for comp_idx, component in enumerate(components[: self.config.max_components]):
                component_ids[row_idx, comp_idx] = self.mappings.component_to_id[str(component)]
                component_doses[row_idx, comp_idx] = float(doses[comp_idx]) if comp_idx < len(doses) else 1.0
                component_mask[row_idx, comp_idx] = 1.0
        fallback_treatment_id = self.mappings.treatment_to_id.get(
            self.config.control_treatment,
            next(iter(self.mappings.treatment_to_id.values())),
        )
        treatment_id = torch.as_tensor(
            [self.mappings.treatment_to_id.get(label, fallback_treatment_id) for label in treatment_labels],
            dtype=torch.long,
            device=self.device,
        )
        return (
            treatment_id,
            torch.as_tensor(component_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(component_doses, dtype=torch.float32, device=self.device),
            torch.as_tensor(component_mask, dtype=torch.float32, device=self.device),
        )

    @torch.no_grad()
    def encode_adata(self, adata: ad.AnnData) -> np.ndarray:
        self._ensure_fitted()
        x = torch.as_tensor(self._expression_from_adata(adata), dtype=torch.float32, device=self.device)
        assert self.model is not None
        self.model.eval()
        return self.model.encode(x).detach().cpu().numpy()

    @torch.no_grad()
    def predict_adata(self, adata: ad.AnnData, treatment: Any, dose: Any = None, covariates: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        self._ensure_fitted()
        x_np = self._expression_from_adata(adata)
        n_obs = x_np.shape[0]
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        treatment_id, component_ids, component_doses, component_mask = self._encode_components_for_prediction(treatment=treatment, dose=dose, n_obs=n_obs)
        covariate_ids = self._encode_covariates_for_prediction(n_obs=n_obs, adata=adata, covariates=covariates)
        assert self.model is not None
        self.model.eval()
        outputs = self.model(
            x=x,
            treatment_id=treatment_id,
            dose=torch.as_tensor(np.maximum(component_doses.cpu().numpy().sum(axis=1), 1.0), dtype=torch.float32, device=self.device),
            covariates=covariate_ids,
            component_ids=component_ids,
            component_doses=component_doses,
            component_mask=component_mask,
        )
        return {
            "x_hat": outputs["x_hat"].detach().cpu().numpy(),
            "z_basal": outputs["z_basal"].detach().cpu().numpy(),
            "z_total": outputs["z_total"].detach().cpu().numpy(),
            "dose_scale": outputs["dose_scale"].detach().cpu().numpy(),
            "component_dose_scale": outputs["component_dose_scale"].detach().cpu().numpy(),
            "perturbation_embedding": outputs["perturbation_embedding"].detach().cpu().numpy(),
        }

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        assert self.model is not None and self.mappings is not None
        payload = {
            "config": asdict(self.config),
            "model_state_dict": self.model.state_dict(),
            "mappings": asdict(self.mappings),
            "history": self.history,
            "epochs_trained": self.epochs_trained,
            "best_val_reconstruction_loss": self.best_val_reconstruction_loss,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RegenAIPTTrainer:
        require_torch()
        payload = torch.load(Path(path), map_location="cpu")
        config = RegenAIPTConfig(**payload["config"])
        trainer = cls(config)
        mappings = RegenAIPTMappings(**payload["mappings"])
        trainer.mappings = mappings
        trainer.model = trainer._initialize_model(mappings)
        trainer.model.load_state_dict(payload["model_state_dict"])
        trainer.history.update(payload.get("history", {}))
        trainer.epochs_trained = int(payload.get("epochs_trained", 0))
        trainer.best_val_reconstruction_loss = payload.get("best_val_reconstruction_loss")
        trainer.model.eval()
        return trainer

__all__ = ["LOSS_COMPONENTS", "RegenAIPTTrainer"]

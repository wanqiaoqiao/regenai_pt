from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd

from ..forward_model import CurrentStateProfile, TransitionPrediction
from .forward_interface import ForwardTransitionModelInterface
from .regenai_pt_config import RegenAIPTConfig
from .regenai_pt_data import RegenAIPTMappings, prepare_round_specific_adata
from .simulation import simulate_two_round_sequence

if TYPE_CHECKING:  # pragma: no cover
    from .regenai_pt_trainer import RegenAIPTTrainer

try:  # pragma: no cover - exercised in torch-enabled environments
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_regenai_pt_stack() -> tuple[Any, Any]:
    try:
        from .regenai_pt_trainer import RegenAIPTTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the regenai_pt forward adapter. "
            "Install torch to use '--transition-model regenai_pt'."
        ) from exc
    return RegenAIPTTrainer, torch


class RegenAIPTForwardAdapter(ForwardTransitionModelInterface):
    def __init__(
        self,
        round1_trainer: RegenAIPTTrainer | None = None,
        round2_trainer: RegenAIPTTrainer | None = None,
    ) -> None:
        self.round1_trainer = round1_trainer
        self.round2_trainer = round2_trainer
        self.config: RegenAIPTConfig | None = getattr(round1_trainer, 'config', None)
        self.gene_names: list[str] = []

    @staticmethod
    def _clone_config(config: RegenAIPTConfig, **overrides: Any) -> RegenAIPTConfig:
        payload = asdict(config)
        payload.update(overrides)
        return RegenAIPTConfig(**payload)

    def _ensure_ready(self) -> None:
        if self.round1_trainer is None:
            raise RuntimeError('RegenAIPTForwardAdapter has not been fitted or loaded')

    def fit(self, adata: ad.AnnData, config: Any | None = None) -> RegenAIPTForwardAdapter:
        RegenAIPTTrainer, _ = _require_regenai_pt_stack()
        if config is None:
            cfg = RegenAIPTConfig()
        else:
            cfg = config if isinstance(config, RegenAIPTConfig) else RegenAIPTConfig(**dict(config))
        round1_adata = prepare_round_specific_adata(
            adata,
            round_number=1,
            treatment_key='treatment',
            dose_key='dose',
            config=cfg,
        )
        round2_adata = prepare_round_specific_adata(
            adata,
            round_number=2,
            treatment_key='treatment',
            dose_key='dose',
            config=cfg,
        )

        round1_config = self._clone_config(cfg, treatment_key='treatment', dose_key='dose')
        round2_covariates = tuple(dict.fromkeys((*cfg.covariate_keys, cfg.round1_treatment_key)))
        round2_config = self._clone_config(
            cfg,
            treatment_key='treatment',
            dose_key='dose',
            covariate_keys=round2_covariates,
        )

        self.round1_trainer = RegenAIPTTrainer(round1_config).fit(round1_adata)
        self.round2_trainer = RegenAIPTTrainer(round2_config).fit(round2_adata)
        self.config = cfg
        self.gene_names = round1_adata.var_names.astype(str).tolist()
        return self

    @staticmethod
    def _metadata_scalars(metadata: dict[str, Any]) -> dict[str, Any]:
        scalars: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                scalars[key] = value
        return scalars

    def _state_to_adata(self, current_adata_or_state: ad.AnnData | CurrentStateProfile | dict[str, Any]) -> tuple[ad.AnnData, dict[str, Any]]:
        if isinstance(current_adata_or_state, ad.AnnData):
            return current_adata_or_state.copy(), {}

        if isinstance(current_adata_or_state, CurrentStateProfile):
            metadata = dict(current_adata_or_state.metadata)
        else:
            metadata = dict(current_adata_or_state)

        if isinstance(metadata.get('adata'), ad.AnnData):
            return metadata['adata'].copy(), self._metadata_scalars(metadata)

        if 'expression' in metadata:
            expression = np.asarray(metadata['expression'], dtype=np.float32)
            if expression.ndim == 1:
                expression = expression[None, :]
            if not self.gene_names:
                raise ValueError('Adapter is missing gene_names and cannot reconstruct AnnData from expression metadata')
            if expression.shape[1] != len(self.gene_names):
                raise ValueError(
                    'Expression vector in CurrentStateProfile.metadata["expression"] has incompatible width '
                    f'({expression.shape[1]} vs expected {len(self.gene_names)})'
                )
            obs = pd.DataFrame([self._metadata_scalars(metadata)] * expression.shape[0])
            var = pd.DataFrame(index=self.gene_names)
            return ad.AnnData(X=expression, obs=obs, var=var), self._metadata_scalars(metadata)

        raise ValueError(
            'RegenAIPTForwardAdapter requires an AnnData object or a CurrentStateProfile with '
            'metadata["adata"] or metadata["expression"] for prediction.'
        )

    @staticmethod
    def _merge_covariates(base: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(base)
        if extra:
            merged.update({key: value for key, value in extra.items() if value is not None})
        return merged

    @staticmethod
    def _resolve_treatment_name(treatment_metadata: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = treatment_metadata.get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]
            if str(value) != '':
                return str(value)
        raise ValueError(f'Missing treatment metadata. Expected one of: {", ".join(keys)}')

    @staticmethod
    def _resolve_dose(treatment_metadata: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = treatment_metadata.get(key)
            if value is not None and str(value) != '':
                return float(value)
        return None

    def _choose_round(self, treatment_metadata: dict[str, Any]) -> int:
        if treatment_metadata.get('round2_treatment') not in (None, ''):
            return 2
        if treatment_metadata.get('round_number') is not None:
            return int(treatment_metadata['round_number'])
        if str(treatment_metadata.get('treatment_role', '')) == 'round2_treatment':
            return 2
        return 1

    def _expression_from_adata(self, adata: ad.AnnData) -> np.ndarray:
        if self.round1_trainer is None:
            raise RuntimeError('Adapter has not been fitted or loaded')
        return self.round1_trainer._expression_from_adata(adata)  # noqa: SLF001

    @staticmethod
    def _bounded(value: float) -> float:
        return float(value / (1.0 + abs(value)))

    def _derive_state_features(
        self,
        x_current: np.ndarray,
        x_future: np.ndarray,
        z_total: np.ndarray,
    ) -> dict[str, float]:
        current_mean = float(np.mean(np.log1p(np.clip(x_current, 0.0, None))))
        future_mean = float(np.mean(np.log1p(np.clip(x_future, 0.0, None))))
        delta_mean = float(np.mean(np.abs(x_future - x_current)))
        future_var = float(np.var(x_future))
        latent_mean = float(np.mean(z_total))
        latent_first = float(np.mean(z_total[:, 0])) if z_total.ndim == 2 and z_total.shape[1] > 0 else latent_mean
        return {
            'target_marker_score': self._bounded(future_mean + max(future_mean - current_mean, 0.0)),
            'stress_score': self._bounded(delta_mean),
            'off_target_score': self._bounded(future_var),
            'pseudotime': 1.0 / (1.0 + np.exp(-latent_first)),
            'fate_probability': 1.0 / (1.0 + np.exp(-latent_mean)),
        }

    def _prediction_to_state(
        self,
        source_adata: ad.AnnData,
        prediction: dict[str, np.ndarray],
        stage: str,
        metadata: dict[str, Any],
    ) -> TransitionPrediction:
        x_current = self._expression_from_adata(source_adata)
        x_future = np.asarray(prediction['x_hat'], dtype=np.float32)
        z_total = np.asarray(prediction['z_total'], dtype=np.float32)
        state_features = self._derive_state_features(x_current=x_current, x_future=x_future, z_total=z_total)
        uncertainty = float(np.std(x_future - x_current) + np.std(z_total, axis=0).mean())
        return TransitionPrediction(
            state_features=state_features,
            stage=stage,
            uncertainty=uncertainty,
            metadata={
                **metadata,
                'expression': x_future,
                'latent': z_total,
                'dose_scale': np.asarray(prediction.get('dose_scale')),
            },
        )

    def predict_expression(
        self,
        current_adata: ad.AnnData,
        treatment_metadata: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        self._ensure_ready()
        round_number = self._choose_round(treatment_metadata)
        base_covariates = dict(treatment_metadata.get('covariates') or {})
        if round_number == 2:
            if self.round2_trainer is None:
                raise RuntimeError('Round2 trainer is unavailable for sequential prediction')
            treatment = self._resolve_treatment_name(treatment_metadata, 'round2_components', 'round2_treatment', 'treatment_name', 'treatment')
            dose = self._resolve_dose(treatment_metadata, 'round2_dose', 'dose')
            if treatment_metadata.get('round1_treatment') is not None:
                base_covariates['round1_treatment'] = str(treatment_metadata['round1_treatment'])
            return self.round2_trainer.predict_adata(current_adata, treatment=treatment, dose=dose, covariates=base_covariates)

        treatment = self._resolve_treatment_name(treatment_metadata, 'round1_components', 'round1_treatment', 'treatment_name', 'treatment')
        dose = self._resolve_dose(treatment_metadata, 'round1_dose', 'dose')
        assert self.round1_trainer is not None
        return self.round1_trainer.predict_adata(current_adata, treatment=treatment, dose=dose, covariates=base_covariates)

    def predict_latent(
        self,
        current_adata: ad.AnnData,
        treatment_metadata: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        prediction = self.predict_expression(current_adata, treatment_metadata)
        return {
            'z_basal': prediction['z_basal'],
            'z_total': prediction['z_total'],
            'dose_scale': prediction['dose_scale'],
            'perturbation_embedding': prediction['perturbation_embedding'],
        }

    def predict(
        self,
        current_adata_or_state: ad.AnnData | CurrentStateProfile | dict[str, Any],
        treatment_metadata: dict[str, Any],
    ) -> TransitionPrediction:
        source_adata, state_metadata = self._state_to_adata(current_adata_or_state)
        merged_metadata = self._merge_covariates(state_metadata, dict(treatment_metadata.get('covariates') or {}))
        round_number = self._choose_round(treatment_metadata)
        prediction = self.predict_expression(source_adata, treatment_metadata)
        stage = 'post_round2' if round_number == 2 else 'post_round1'
        return self._prediction_to_state(source_adata, prediction, stage=stage, metadata=merged_metadata)

    def simulate_sequence(
        self,
        current_adata_or_state: ad.AnnData | CurrentStateProfile | dict[str, Any],
        treatment_sequence: str | tuple[str, str] | list[str] | dict[str, Any],
        covariates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()
        source_adata, state_metadata = self._state_to_adata(current_adata_or_state)
        if self.round2_trainer is None:
            raise RuntimeError('Round2 trainer is unavailable for sequential simulation')

        if isinstance(treatment_sequence, str):
            if '->' not in treatment_sequence:
                raise ValueError('String treatment_sequence must look like "A->B"')
            round1_treatment, round2_treatment = [part.strip() for part in treatment_sequence.split('->', 1)]
        elif isinstance(treatment_sequence, dict):
            round1_treatment = self._resolve_treatment_name(treatment_sequence, 'round1_components', 'round1_treatment')
            round2_treatment = self._resolve_treatment_name(treatment_sequence, 'round2_components', 'round2_treatment')
        else:
            if len(treatment_sequence) != 2:
                raise ValueError('Treatment sequence must contain exactly two treatments')
            round1_treatment, round2_treatment = [str(value) for value in treatment_sequence]

        round1_dose = None
        round2_dose = None
        if isinstance(treatment_sequence, dict):
            round1_dose = treatment_sequence.get('round1_dose')
            round2_dose = treatment_sequence.get('round2_dose')

        sequence_prediction = simulate_two_round_sequence(
            {'round1': self.round1_trainer, 'round2': self.round2_trainer},
            source_adata,
            round1_treatment=round1_treatment,
            round2_treatment=round2_treatment,
            round1_dose=round1_dose,
            round2_dose=round2_dose,
            covariates=self._merge_covariates(state_metadata, covariates),
        )
        predicted_state = self._prediction_to_state(
            source_adata,
            sequence_prediction,
            stage='post_round2',
            metadata={
                **state_metadata,
                **(covariates or {}),
                'round1_treatment': round1_treatment,
                'round2_treatment': round2_treatment,
            },
        )
        return {
            **sequence_prediction,
            'predicted_state': predicted_state,
        }

    def predict_future_state(
        self,
        current_state: CurrentStateProfile | dict[str, Any],
        round1_treatment: str,
        round2_treatment: str | None = None,
        iPSC_line: str | None = None,
    ) -> TransitionPrediction:
        metadata: dict[str, Any] = {}
        if isinstance(current_state, CurrentStateProfile):
            metadata.update(current_state.metadata)
        if iPSC_line is not None:
            metadata['iPSC_line'] = iPSC_line

        if round2_treatment is not None:
            result = self.simulate_sequence(
                current_state,
                {'round1_treatment': round1_treatment, 'round2_treatment': round2_treatment},
                covariates=metadata,
            )
            return result['predicted_state']

        return self.predict(
            current_state,
            {
                'round_number': 1,
                'round1_treatment': round1_treatment,
                'covariates': metadata,
            },
        )

    @staticmethod
    def _serialize_trainer(trainer: RegenAIPTTrainer | None) -> dict[str, Any] | None:
        if trainer is None:
            return None
        if trainer.model is None or trainer.mappings is None:
            raise RuntimeError('Cannot serialize an unfitted RegenAIPTTrainer')
        return {
            'config': asdict(trainer.config),
            'model_state_dict': {key: value.detach().cpu() for key, value in trainer.model.state_dict().items()},
            'mappings': {
                'treatment_to_id': trainer.mappings.treatment_to_id,
                'id_to_treatment': trainer.mappings.id_to_treatment,
                'component_to_id': trainer.mappings.component_to_id,
                'id_to_component': trainer.mappings.id_to_component,
                'covariate_to_id': trainer.mappings.covariate_to_id,
                'id_to_covariate': trainer.mappings.id_to_covariate,
                'gene_names': trainer.mappings.gene_names,
                'input_dim': trainer.mappings.input_dim,
                'max_components': trainer.mappings.max_components,
            },
            'history': trainer.history,
            'epochs_trained': trainer.epochs_trained,
            'best_val_reconstruction_loss': trainer.best_val_reconstruction_loss,
        }

    @staticmethod
    def _deserialize_trainer(payload: dict[str, Any] | None) -> RegenAIPTTrainer | None:
        if payload is None:
            return None
        RegenAIPTTrainer, _ = _require_regenai_pt_stack()
        trainer = RegenAIPTTrainer(RegenAIPTConfig(**payload['config']))
        mappings = RegenAIPTMappings(**payload['mappings'])
        trainer.mappings = mappings
        trainer.model = trainer._initialize_model(mappings)
        trainer.model.load_state_dict(payload['model_state_dict'])
        trainer.history = payload.get('history', trainer.history)
        trainer.epochs_trained = int(payload.get('epochs_trained', 0))
        trainer.best_val_reconstruction_loss = payload.get('best_val_reconstruction_loss')
        trainer.model.eval()
        return trainer

    def save(self, path: str | Path) -> None:
        _, torch_module = _require_regenai_pt_stack()
        payload = {
            'config': None if self.config is None else asdict(self.config),
            'gene_names': self.gene_names,
            'round1_trainer': self._serialize_trainer(self.round1_trainer),
            'round2_trainer': self._serialize_trainer(self.round2_trainer),
        }
        torch_module.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RegenAIPTForwardAdapter:
        _, torch_module = _require_regenai_pt_stack()
        payload = torch_module.load(Path(path), map_location='cpu')
        adapter = cls(
            round1_trainer=cls._deserialize_trainer(payload.get('round1_trainer')),
            round2_trainer=cls._deserialize_trainer(payload.get('round2_trainer')),
        )
        config_payload = payload.get('config')
        adapter.config = None if config_payload is None else RegenAIPTConfig(**config_payload)
        adapter.gene_names = [str(name) for name in payload.get('gene_names', [])]
        return adapter


__all__ = ['RegenAIPTForwardAdapter']

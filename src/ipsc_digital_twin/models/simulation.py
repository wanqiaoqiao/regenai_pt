from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


def _ensure_predictor(model: Any) -> Any:
    if hasattr(model, 'predict_adata'):
        return model
    raise TypeError('model must provide a predict_adata(...) method or be a round-mapped predictor bundle')



def _prediction_to_adata(prediction: dict[str, np.ndarray], template_adata: ad.AnnData) -> ad.AnnData:
    obs = template_adata.obs.copy()
    var = pd.DataFrame(index=template_adata.var_names.astype(str))
    predicted = ad.AnnData(X=np.asarray(prediction['x_hat'], dtype=np.float32), obs=obs, var=var)
    for layer_name in template_adata.layers:
        predicted.layers[layer_name] = np.asarray(prediction['x_hat'], dtype=np.float32)
    return predicted



def simulate_round1(model: Any, current_adata: ad.AnnData, round1_treatment: Any, dose: Any = None, covariates: dict[str, Any] | None = None) -> dict[str, Any]:
    predictor = _ensure_predictor(model)
    pred = predictor.predict_adata(current_adata, treatment=round1_treatment, dose=dose, covariates=covariates)
    predicted_adata = _prediction_to_adata(pred, current_adata)
    predicted_adata.obs['round'] = 1
    predicted_adata.obs['time_point'] = 'post_round1'
    predicted_adata.obs['round1_treatment'] = '+'.join(round1_treatment) if isinstance(round1_treatment, (list, tuple)) else str(round1_treatment)
    predicted_adata.obs['round1_components'] = [list(round1_treatment) if isinstance(round1_treatment, (list, tuple)) else [str(round1_treatment)]] * predicted_adata.n_obs
    pred['predicted_adata'] = predicted_adata
    return pred



def simulate_round2(model: Any, post_round1_adata_or_prediction: Any, round2_treatment: Any, dose: Any = None, covariates: dict[str, Any] | None = None) -> dict[str, Any]:
    predictor = _ensure_predictor(model)
    if isinstance(post_round1_adata_or_prediction, ad.AnnData):
        post_round1_adata = post_round1_adata_or_prediction
    elif isinstance(post_round1_adata_or_prediction, dict) and 'predicted_adata' in post_round1_adata_or_prediction:
        post_round1_adata = post_round1_adata_or_prediction['predicted_adata']
    else:
        raise TypeError('post_round1_adata_or_prediction must be an AnnData or round1 prediction payload')
    pred = predictor.predict_adata(post_round1_adata, treatment=round2_treatment, dose=dose, covariates=covariates)
    predicted_adata = _prediction_to_adata(pred, post_round1_adata)
    predicted_adata.obs['round'] = 2
    predicted_adata.obs['time_point'] = 'post_round2'
    predicted_adata.obs['round2_treatment'] = '+'.join(round2_treatment) if isinstance(round2_treatment, (list, tuple)) else str(round2_treatment)
    predicted_adata.obs['round2_components'] = [list(round2_treatment) if isinstance(round2_treatment, (list, tuple)) else [str(round2_treatment)]] * predicted_adata.n_obs
    pred['predicted_adata'] = predicted_adata
    return pred



def simulate_two_round_sequence(
    model: Any,
    current_adata: ad.AnnData,
    round1_treatment: Any,
    round2_treatment: Any,
    round1_dose: Any = None,
    round2_dose: Any = None,
    covariates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(model, dict):
        round1_model = _ensure_predictor(model['round1'])
        round2_model = _ensure_predictor(model['round2'])
    else:
        round1_model = _ensure_predictor(model)
        round2_model = round1_model

    round1_prediction = simulate_round1(
        round1_model,
        current_adata=current_adata,
        round1_treatment=round1_treatment,
        dose=round1_dose,
        covariates=covariates,
    )
    round2_covariates = dict(covariates or {})
    round1_label = '+'.join(round1_treatment) if isinstance(round1_treatment, (list, tuple)) else str(round1_treatment)
    round2_covariates.setdefault('round1_treatment', round1_label)
    round2_prediction = simulate_round2(
        round2_model,
        post_round1_adata_or_prediction=round1_prediction,
        round2_treatment=round2_treatment,
        dose=round2_dose,
        covariates=round2_covariates,
    )
    return {
        'round1': round1_prediction,
        'round2': round2_prediction,
        'x_hat': round2_prediction['x_hat'],
        'z_basal': round2_prediction['z_basal'],
        'z_total': round2_prediction['z_total'],
        'dose_scale': round2_prediction['dose_scale'],
        'component_dose_scale': round2_prediction.get('component_dose_scale'),
        'perturbation_embedding': round2_prediction['perturbation_embedding'],
        'predicted_adata': round2_prediction['predicted_adata'],
    }

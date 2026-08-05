from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=True)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = math.sqrt(float((x_center**2).sum())) * math.sqrt(float((y_center**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((x_center * y_center).sum() / denom)


def compute_spearman_ranking_correlation(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    group_col: str,
    score_col_pred: str,
    score_col_obs: str,
) -> float:
    merged = predictions[[group_col, score_col_pred]].merge(
        observed[[group_col, score_col_obs]], on=group_col, how="inner"
    )
    if merged.empty:
        return float("nan")
    rp = _rank_series(merged[score_col_pred].astype(float))
    ro = _rank_series(merged[score_col_obs].astype(float))
    return _pearson(rp.to_numpy(), ro.to_numpy())


def compute_rmse(y_pred: np.ndarray | pd.Series, y_true: np.ndarray | pd.Series) -> float:
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    if yp.shape != yt.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def compute_topk_overlap(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    group_col: str,
    score_col_pred: str,
    score_col_obs: str,
    k: int = 10,
) -> float:
    pred_top = (
        predictions[[group_col, score_col_pred]]
        .sort_values(score_col_pred, ascending=False)
        .head(int(k))[group_col]
        .astype(str)
    )
    obs_top = (
        observed[[group_col, score_col_obs]]
        .sort_values(score_col_obs, ascending=False)
        .head(int(k))[group_col]
        .astype(str)
    )
    if len(pred_top) == 0 or len(obs_top) == 0:
        return float("nan")
    inter = len(set(pred_top).intersection(set(obs_top)))
    return float(inter / max(1, int(k)))


def compute_direction_accuracy(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    group_col: str,
    score_col_pred: str,
    score_col_obs: str,
    control_label: str = "control",
) -> float:
    p = predictions[[group_col, score_col_pred]].copy()
    o = observed[[group_col, score_col_obs]].copy()

    if control_label not in set(p[group_col].astype(str)) or control_label not in set(o[group_col].astype(str)):
        return float("nan")

    p_control = float(p[p[group_col].astype(str) == control_label][score_col_pred].mean())
    o_control = float(o[o[group_col].astype(str) == control_label][score_col_obs].mean())

    merged = p.merge(o, on=group_col, how="inner")
    merged = merged[merged[group_col].astype(str) != control_label]
    if merged.empty:
        return float("nan")

    pred_improve = (merged[score_col_pred].astype(float) - p_control) > 0
    obs_improve = (merged[score_col_obs].astype(float) - o_control) > 0
    return float((pred_improve == obs_improve).mean())


def compute_stress_risk_detection_accuracy(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    group_col: str,
    pred_risk_col: str = "pred_stress_risk",
    label_col: str = "stress_risk_label",
) -> float:
    if pred_risk_col not in predictions.columns or label_col not in observed.columns:
        return float("nan")

    merged = predictions[[group_col, pred_risk_col]].merge(observed[[group_col, label_col]], on=group_col, how="inner")
    if merged.empty:
        return float("nan")

    pred = merged[pred_risk_col].astype(int)
    true = merged[label_col].astype(int)
    return float((pred == true).mean())

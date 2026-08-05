from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

DEFAULT_STATE_FEATURES = (
    "target_marker_score",
    "stress_score",
    "off_target_score",
    "pseudotime",
    "fate_probability",
)
DEFAULT_CONTEXT_COLS = (
    "experiment_id",
    "sample_id",
    "iPSC_line",
    "replicate",
    "batch",
    "barcode_group",
)


@dataclass
class CurrentStateProfile:
    state_features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetStateProfile:
    state_features: dict[str, float]
    feature_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionPrediction:
    state_features: dict[str, float]
    stage: str
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)



def _available_state_features(adata: ad.AnnData, state_features: list[str] | None = None) -> list[str]:
    requested = list(state_features) if state_features is not None else list(DEFAULT_STATE_FEATURES)
    features = [feature for feature in requested if feature in adata.obs.columns]
    if not features:
        raise ValueError(
            "No usable state feature columns found in adata.obs. "
            f"Looked for: {', '.join(requested)}"
        )
    return features



def _normalize_series(values: dict[str, Any] | pd.Series | CurrentStateProfile) -> pd.Series:
    if isinstance(values, CurrentStateProfile):
        payload = values.state_features
    elif isinstance(values, pd.Series):
        payload = values.to_dict()
    else:
        payload = dict(values)
    return pd.Series(payload, dtype=float)



def build_population_state_table(
    adata: ad.AnnData,
    state_features: list[str] | None = None,
    context_cols: list[str] | None = None,
) -> pd.DataFrame:
    features = _available_state_features(adata, state_features)
    group_cols = [col for col in (context_cols or list(DEFAULT_CONTEXT_COLS)) if col in adata.obs.columns]
    required_cols = ["time_point", "round1_treatment", "round2_treatment", *group_cols, *features]
    frame = adata.obs[[col for col in required_cols if col in adata.obs.columns]].copy()

    if not group_cols:
        frame["__global_group"] = "global"
        group_cols = ["__global_group"]

    grouped = (
        frame.groupby(["time_point", "round1_treatment", "round2_treatment", *group_cols], dropna=False)[features]
        .mean()
        .reset_index()
    )

    if "stage_label" in adata.obs.columns:
        stage_props = (
            adata.obs[["time_point", "round1_treatment", "round2_treatment", *group_cols, "stage_label"]]
            .assign(_count=1)
            .pivot_table(
                index=["time_point", "round1_treatment", "round2_treatment", *group_cols],
                columns="stage_label",
                values="_count",
                aggfunc="sum",
                fill_value=0,
            )
        )
        if not stage_props.empty:
            stage_props = stage_props.div(stage_props.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            stage_props.columns = [f"stage_prop_{column}" for column in stage_props.columns]
            grouped = grouped.merge(stage_props.reset_index(), on=["time_point", "round1_treatment", "round2_treatment", *group_cols], how="left")

    return grouped



def build_transition_table(
    adata: ad.AnnData,
    state_features: list[str] | None = None,
    context_cols: list[str] | None = None,
) -> pd.DataFrame:
    population = build_population_state_table(adata=adata, state_features=state_features, context_cols=context_cols)
    state_cols = [
        column
        for column in population.columns
        if column in _available_state_features(adata, state_features) or column.startswith("stage_prop_")
    ]
    context = [column for column in (context_cols or list(DEFAULT_CONTEXT_COLS)) if column in population.columns]
    if "__global_group" in population.columns and "__global_group" not in context:
        context.append("__global_group")

    intermediate = population[population["time_point"] == "intermediate"].copy()
    post_round1 = population[population["time_point"] == "post_round1"].copy()
    post_round2 = population[population["time_point"] == "post_round2"].copy()

    if intermediate.empty or post_round1.empty:
        raise ValueError("Need both intermediate and post_round1 population states to build transition table")

    round1 = post_round1.merge(
        intermediate[[*context, *state_cols]],
        on=context,
        how="left",
        suffixes=("_future", "_current"),
    )
    round1["transition_stage"] = "round1"

    frames = [round1]
    if not post_round2.empty:
        round2 = post_round2.merge(
            post_round1[[*context, "round1_treatment", *state_cols]],
            on=[*context, "round1_treatment"],
            how="left",
            suffixes=("_future", "_current"),
        )
        round2["transition_stage"] = "round2"
        frames.append(round2)

    transition_table = pd.concat(frames, ignore_index=True, sort=False)
    missing_current = [feature for feature in state_cols if f"{feature}_current" not in transition_table.columns]
    if missing_current:
        raise ValueError(f"Failed to build transition table for state features: {', '.join(missing_current)}")
    return transition_table


class ForwardTransitionModel:
    def __init__(self, state_features: list[str] | None = None) -> None:
        self.state_features = state_features or []
        self.round1_effects: dict[tuple[str | None, str], dict[str, Any]] = {}
        self.round1_global_effects: dict[str, dict[str, Any]] = {}
        self.round2_effects: dict[tuple[str | None, str, str], dict[str, Any]] = {}
        self.round2_global_effects: dict[tuple[str, str], dict[str, Any]] = {}
        self.reference_state: dict[str, float] = {}
        self.fitted = False

    def fit(self, adata: ad.AnnData, state_features: list[str] | None = None) -> ForwardTransitionModel:
        transition_table = build_transition_table(adata=adata, state_features=state_features)
        return self.fit_transition_table(transition_table)

    def fit_transition_table(self, transition_table: pd.DataFrame) -> ForwardTransitionModel:
        current_cols = [column for column in transition_table.columns if column.endswith("_current")]
        if not current_cols:
            raise ValueError("transition_table must include *_current state columns")

        self.state_features = [column.removesuffix("_current") for column in current_cols]
        self.reference_state = {
            feature: float(transition_table[f"{feature}_current"].mean())
            for feature in self.state_features
        }

        for feature in self.state_features:
            transition_table[f"delta_{feature}"] = (
                transition_table[f"{feature}_future"] - transition_table[f"{feature}_current"]
            )

        delta_cols = [f"delta_{feature}" for feature in self.state_features]
        group_line = "iPSC_line" if "iPSC_line" in transition_table.columns else None

        round1 = transition_table[transition_table["transition_stage"] == "round1"].copy()
        if round1.empty:
            raise ValueError("No round1 transitions available for forward model training")

        if group_line is not None:
            grouped = round1.groupby([group_line, "round1_treatment"], dropna=False)[delta_cols]
            for key, frame in grouped:
                line, treatment = key
                self.round1_effects[(str(line), str(treatment))] = {
                    "mean_delta": frame.mean().to_dict(),
                    "uncertainty": float(frame.std(ddof=0).fillna(0.0).mean()),
                    "n": int(len(frame)),
                }

        for treatment, frame in round1.groupby("round1_treatment", dropna=False)[delta_cols]:
            self.round1_global_effects[str(treatment)] = {
                "mean_delta": frame.mean().to_dict(),
                "uncertainty": float(frame.std(ddof=0).fillna(0.0).mean()),
                "n": int(len(frame)),
            }

        round2 = transition_table[transition_table["transition_stage"] == "round2"].copy()
        if not round2.empty:
            if group_line is not None:
                grouped = round2.groupby([group_line, "round1_treatment", "round2_treatment"], dropna=False)[delta_cols]
                for key, frame in grouped:
                    line, round1_treatment, round2_treatment = key
                    self.round2_effects[(str(line), str(round1_treatment), str(round2_treatment))] = {
                        "mean_delta": frame.mean().to_dict(),
                        "uncertainty": float(frame.std(ddof=0).fillna(0.0).mean()),
                        "n": int(len(frame)),
                    }

            for key, frame in round2.groupby(["round1_treatment", "round2_treatment"], dropna=False)[delta_cols]:
                round1_treatment, round2_treatment = key
                self.round2_global_effects[(str(round1_treatment), str(round2_treatment))] = {
                    "mean_delta": frame.mean().to_dict(),
                    "uncertainty": float(frame.std(ddof=0).fillna(0.0).mean()),
                    "n": int(len(frame)),
                }

        self.fitted = True
        return self

    def _effect_lookup(self, line: str | None, round1_treatment: str, round2_treatment: str | None = None) -> dict[str, Any]:
        if round2_treatment is None:
            if line is not None and (line, round1_treatment) in self.round1_effects:
                return self.round1_effects[(line, round1_treatment)]
            return self.round1_global_effects.get(round1_treatment, {"mean_delta": {}, "uncertainty": 1.0, "n": 0})

        if line is not None and (line, round1_treatment, round2_treatment) in self.round2_effects:
            return self.round2_effects[(line, round1_treatment, round2_treatment)]
        return self.round2_global_effects.get((round1_treatment, round2_treatment), {"mean_delta": {}, "uncertainty": 1.0, "n": 0})

    def predict_future_state(
        self,
        current_state: dict[str, Any] | pd.Series | CurrentStateProfile,
        round1_treatment: str,
        round2_treatment: str | None = None,
        iPSC_line: str | None = None,
    ) -> TransitionPrediction:
        if not self.fitted:
            raise RuntimeError("ForwardTransitionModel must be fitted before prediction")

        current = _normalize_series(current_state)
        for feature, fallback in self.reference_state.items():
            if feature not in current.index or pd.isna(current[feature]):
                current[feature] = fallback
        current = current[self.state_features].astype(float)

        round1_effect = self._effect_lookup(iPSC_line, str(round1_treatment))
        future = current.copy()
        for feature in self.state_features:
            future[feature] = future[feature] + float(round1_effect["mean_delta"].get(f"delta_{feature}", 0.0))
        total_uncertainty = float(round1_effect.get("uncertainty", 0.0))
        stage = "post_round1"

        if round2_treatment is not None:
            round2_effect = self._effect_lookup(iPSC_line, str(round1_treatment), str(round2_treatment))
            for feature in self.state_features:
                future[feature] = future[feature] + float(round2_effect["mean_delta"].get(f"delta_{feature}", 0.0))
            total_uncertainty += float(round2_effect.get("uncertainty", 0.0))
            stage = "post_round2"

        return TransitionPrediction(
            state_features={feature: float(future[feature]) for feature in self.state_features},
            stage=stage,
            uncertainty=total_uncertainty,
            metadata={
                "round1_treatment": str(round1_treatment),
                "round2_treatment": None if round2_treatment is None else str(round2_treatment),
                "iPSC_line": iPSC_line,
            },
        )

    def save(self, path: str | Path) -> None:
        with open(Path(path), "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> ForwardTransitionModel:
        with open(Path(path), "rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Artifact did not contain a ForwardTransitionModel")
        return model

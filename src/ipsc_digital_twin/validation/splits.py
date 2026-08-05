from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def _holdout_by_group(df: pd.DataFrame, group_col: str) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    groups = [g for g in df[group_col].dropna().unique().tolist()]
    for g in groups:
        test = df[df[group_col] == g].copy()
        train = df[df[group_col] != g].copy()
        if train.empty or test.empty:
            continue
        yield train, test, str(g)


def holdout_by_treatment(df: pd.DataFrame, treatment_col: str = "round1_treatment") -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    return _holdout_by_group(df, treatment_col)


def holdout_by_ipsc_line(df: pd.DataFrame, line_col: str = "iPSC_line") -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    return _holdout_by_group(df, line_col)


def holdout_by_treatment_sequence(
    df: pd.DataFrame,
    sequence_col: str = "treatment_sequence",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    return _holdout_by_group(df, sequence_col)


def holdout_by_replicate(df: pd.DataFrame, replicate_col: str = "replicate") -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    return _holdout_by_group(df, replicate_col)

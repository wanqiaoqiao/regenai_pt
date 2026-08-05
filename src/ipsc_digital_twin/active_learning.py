from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_library(treatment_library: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    lib = pd.DataFrame(treatment_library) if not isinstance(treatment_library, pd.DataFrame) else treatment_library.copy()
    if "treatment_name" not in lib.columns:
        raise ValueError("treatment_library must include treatment_name")
    if "active" not in lib.columns:
        lib["active"] = True
    if "pathway" not in lib.columns:
        lib["pathway"] = "unknown"
    return lib


def _parse_sequence(seq: str) -> tuple[str, str]:
    if "->" in str(seq):
        a, b = str(seq).split("->", 1)
        return a, b
    return str(seq), ""


def _expand_replicates(df: pd.DataFrame, n_replicates: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for r in range(1, n_replicates + 1):
            out = row.to_dict()
            out["replicate"] = r
            rows.append(out)
    return pd.DataFrame(rows)


def _suggest_plate_map(doe: pd.DataFrame) -> pd.DataFrame:
    # 2 x 48 wells (A1-H6 per plate)
    wells = [f"{row}{col}" for col in range(1, 7) for row in "ABCDEFGH"]
    slots = []
    for plate in ["plate_1", "plate_2"]:
        for w in wells:
            slots.append((plate, w))
    n = min(len(doe), len(slots))
    mapped = doe.head(n).copy().reset_index(drop=True)
    mapped["plate"] = [slots[i][0] for i in range(n)]
    mapped["well"] = [slots[i][1] for i in range(n)]
    return mapped


def recommend_next_doe(
    scored_table: pd.DataFrame,
    treatment_library: list[dict[str, Any]] | pd.DataFrame,
    max_conditions: int = 48,
    n_replicates: int = 2,
    exploitation_fraction: float = 0.4,
    exploration_fraction: float = 0.3,
    diversity_fraction: float = 0.2,
    control_fraction: float = 0.1,
) -> dict[str, Any]:
    if "treatment_sequence" not in scored_table.columns:
        raise ValueError("scored_table must include treatment_sequence")
    if "final_score" not in scored_table.columns:
        raise ValueError("scored_table must include final_score")

    lib = _normalize_library(treatment_library)
    active_names = set(lib.loc[lib["active"].astype(bool), "treatment_name"].astype(str))

    work = scored_table.copy()
    work["round1_treatment"] = work["treatment_sequence"].astype(str).map(lambda s: _parse_sequence(s)[0])
    work["round2_treatment"] = work["treatment_sequence"].astype(str).map(lambda s: _parse_sequence(s)[1])
    work = work[
        work["round1_treatment"].isin(active_names)
        & work["round2_treatment"].isin(active_names)
    ].copy()

    if "uncertainty" not in work.columns:
        work["uncertainty"] = 0.0

    if "stress_score" not in work.columns:
        work["stress_score"] = 0.0
    if "off_target_score" not in work.columns:
        work["off_target_score"] = 0.0

    # Allocation on unique conditions before replicate expansion
    total_unique = max(1, int(round(max_conditions / max(1, n_replicates))))
    n_control = max(1, int(round(total_unique * control_fraction)))
    n_exploit = max(1, int(round(total_unique * exploitation_fraction)))
    n_explore = max(1, int(round(total_unique * exploration_fraction)))
    n_diverse = max(1, int(round(total_unique * diversity_fraction)))

    # Adjust to exact budget
    allocated = n_control + n_exploit + n_explore + n_diverse
    while allocated > total_unique:
        for key in ["n_explore", "n_diverse", "n_exploit", "n_control"]:
            if locals()[key] > 1 and allocated > total_unique:
                locals()[key] -= 1
                allocated -= 1
    while allocated < total_unique:
        n_exploit += 1
        allocated += 1

    selected = []
    seen = set()

    # Controls: control, best, worst, standard protocol
    controls = []
    for special in ["control->control", "standard->standard"]:
        if special in set(work["treatment_sequence"].astype(str)):
            controls.append(work[work["treatment_sequence"] == special].iloc[0])
    if not work.empty:
        controls.append(work.sort_values("final_score", ascending=False).iloc[0])
        controls.append(work.sort_values("final_score", ascending=True).iloc[0])

    for row in controls:
        seq = str(row["treatment_sequence"])
        if seq not in seen:
            selected.append((row, "control"))
            seen.add(seq)
        if len([x for x in selected if x[1] == "control"]) >= n_control:
            break

    # Exploitation
    for _, row in work.sort_values("final_score", ascending=False).iterrows():
        seq = str(row["treatment_sequence"])
        if seq in seen:
            continue
        selected.append((row, "exploitation_high_final_score"))
        seen.add(seq)
        if len([x for x in selected if x[1].startswith("exploitation")]) >= n_exploit:
            break

    # Exploration: high uncertainty + safe
    explore_pool = work[(work["stress_score"] <= 0.6) & (work["off_target_score"] <= 0.6)]
    for _, row in explore_pool.sort_values("uncertainty", ascending=False).iterrows():
        seq = str(row["treatment_sequence"])
        if seq in seen:
            continue
        selected.append((row, "exploration_high_uncertainty_safe"))
        seen.add(seq)
        if len([x for x in selected if x[1].startswith("exploration")]) >= n_explore:
            break

    # Diversity: maximize distinct pathways using round2 treatment pathway
    path_map = {str(r["treatment_name"]): str(r.get("pathway", "unknown")) for _, r in lib.iterrows()}
    diverse_count = 0
    used_pathways: set[str] = set()
    for _, row in work.sort_values("final_score", ascending=False).iterrows():
        seq = str(row["treatment_sequence"])
        if seq in seen:
            continue
        pwy = path_map.get(str(row["round2_treatment"]), "unknown")
        if pwy in used_pathways and len(used_pathways) >= 2:
            continue
        selected.append((row, f"diversity_pathway_{pwy}"))
        seen.add(seq)
        used_pathways.add(pwy)
        diverse_count += 1
        if diverse_count >= n_diverse:
            break

    # Fill any remainder by score
    for _, row in work.sort_values("final_score", ascending=False).iterrows():
        seq = str(row["treatment_sequence"])
        if seq in seen:
            continue
        selected.append((row, "fill_best_remaining"))
        seen.add(seq)
        if len(selected) >= total_unique:
            break

    chosen_rows = []
    for idx, (row, why) in enumerate(selected[:total_unique], start=1):
        chosen_rows.append(
            {
                "condition_id": f"COND_{idx:03d}",
                "treatment_sequence": str(row["treatment_sequence"]),
                "round1_treatment": str(row["round1_treatment"]),
                "round2_treatment": str(row["round2_treatment"]),
                "final_score": float(row["final_score"]),
                "uncertainty": float(row.get("uncertainty", 0.0)),
                "rationale": why,
            }
        )

    unique_doe = pd.DataFrame(chosen_rows)
    doe = _expand_replicates(unique_doe, n_replicates=n_replicates)
    doe = doe.head(max_conditions).reset_index(drop=True)

    plate_map = _suggest_plate_map(doe)

    top_lines = [
        f"- {r.treatment_sequence} ({r.rationale}), score={r.final_score:.3f}, repl={int(r.replicate)}"
        for r in doe.head(20).itertuples(index=False)
    ]
    md = "\n".join(
        [
            "# DOE Recommendation",
            "",
            f"Requested max conditions: {max_conditions}",
            f"Replicates per condition: {n_replicates}",
            "",
            "## Selection Mix",
            f"- Exploitation fraction: {exploitation_fraction}",
            f"- Exploration fraction: {exploration_fraction}",
            f"- Diversity fraction: {diversity_fraction}",
            f"- Control fraction: {control_fraction}",
            "",
            "## Selected Conditions (preview)",
            *(top_lines if top_lines else ["- none"]),
            "",
        ]
    )

    return {
        "doe_table": doe,
        "plate_map": plate_map,
        "markdown_report": md,
    }

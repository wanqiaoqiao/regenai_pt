from __future__ import annotations

import pandas as pd

from ipsc_digital_twin.active_learning import recommend_next_doe


def _library() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"treatment_name": "control", "active": True, "pathway": "control"},
            {"treatment_name": "standard", "active": True, "pathway": "standard"},
            {"treatment_name": "A", "active": True, "pathway": "WNT"},
            {"treatment_name": "B", "active": True, "pathway": "TGFb"},
            {"treatment_name": "C", "active": True, "pathway": "FGF"},
            {"treatment_name": "D", "active": True, "pathway": "BMP"},
            {"treatment_name": "INACTIVE", "active": False, "pathway": "WNT"},
        ]
    )


def _scored() -> pd.DataFrame:
    rows = [
        {"treatment_sequence": "control->control", "final_score": 0.1, "uncertainty": 0.1, "stress_score": 0.1, "off_target_score": 0.1},
        {"treatment_sequence": "standard->standard", "final_score": 0.2, "uncertainty": 0.2, "stress_score": 0.2, "off_target_score": 0.2},
        {"treatment_sequence": "A->B", "final_score": 0.9, "uncertainty": 0.2, "stress_score": 0.2, "off_target_score": 0.2},
        {"treatment_sequence": "B->C", "final_score": 0.8, "uncertainty": 0.8, "stress_score": 0.2, "off_target_score": 0.2},
        {"treatment_sequence": "C->D", "final_score": 0.7, "uncertainty": 0.7, "stress_score": 0.3, "off_target_score": 0.2},
        {"treatment_sequence": "D->A", "final_score": 0.6, "uncertainty": 0.9, "stress_score": 0.2, "off_target_score": 0.2},
        {"treatment_sequence": "A->INACTIVE", "final_score": 0.95, "uncertainty": 0.5, "stress_score": 0.1, "off_target_score": 0.1},
    ]
    return pd.DataFrame(rows)


def test_recommend_next_doe_outputs_requested_size_and_replicates() -> None:
    out = recommend_next_doe(
        scored_table=_scored(),
        treatment_library=_library(),
        max_conditions=12,
        n_replicates=2,
    )
    doe = out["doe_table"]
    assert len(doe) == 12
    assert set(doe["replicate"]) == {1, 2}


def test_includes_controls_and_diversity_and_no_inactive() -> None:
    out = recommend_next_doe(
        scored_table=_scored(),
        treatment_library=_library(),
        max_conditions=12,
        n_replicates=2,
    )
    doe = out["doe_table"]

    seqs = set(doe["treatment_sequence"].astype(str))
    assert "control->control" in seqs

    # No inactive treatment in selected round2
    assert "INACTIVE" not in set(doe["round2_treatment"].astype(str))

    # Diversity via rationale pathway tags
    diversity_tags = [r for r in doe["rationale"].astype(str).tolist() if r.startswith("diversity_pathway_")]
    assert len(diversity_tags) > 0


def test_plate_map_and_markdown_present() -> None:
    out = recommend_next_doe(
        scored_table=_scored(),
        treatment_library=_library(),
        max_conditions=20,
        n_replicates=2,
    )
    plate = out["plate_map"]
    md = out["markdown_report"]
    assert len(plate) > 0
    assert {"plate", "well"}.issubset(set(plate.columns))
    assert "DOE Recommendation" in md

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.treatments import (
    generate_treatment_mechanism_summary,
    load_treatment_library,
    map_obs_treatments_to_library,
    validate_treatment_library,
)

TREATMENT_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "examples" / "treatment_library.yaml"


@pytest.fixture
def mini_adata() -> ad.AnnData:
    x = np.ones((4, 3), dtype=float)
    obs = pd.DataFrame(
        {
            "round1_treatment": ["CHIR99021", "Activin A", "UNKNOWN_DRUG", "CHIR99021"],
            "round2_treatment": ["Activin A", "BMP4", "BMP4", "UNKNOWN_2"],
        }
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_valid_library_loads() -> None:
    library = load_treatment_library(TREATMENT_LIBRARY_PATH)
    assert len(library) >= 2
    assert library[0]["treatment_id"]


def test_missing_required_fields_fail() -> None:
    bad = [
        {
            "treatment_id": "X",
            "treatment_name": "Bad",
            "treatment_type": "small_molecule",
        }
    ]
    with pytest.raises(ValueError):
        validate_treatment_library(bad)


def test_obs_treatment_mapping_and_unknown_report(mini_adata) -> None:
    library = load_treatment_library(TREATMENT_LIBRARY_PATH)

    with pytest.warns(UserWarning):
        report = map_obs_treatments_to_library(mini_adata, library)

    assert "round1_treatment_id" in mini_adata.obs.columns
    assert "round2_treatment_id" in mini_adata.obs.columns
    assert "UNKNOWN_DRUG" in report["unknown_treatments"]
    assert "UNKNOWN_2" in report["unknown_treatments"]


def test_generate_mechanism_summary() -> None:
    library = load_treatment_library(TREATMENT_LIBRARY_PATH)
    summary = generate_treatment_mechanism_summary(library)
    assert len(summary) > 0
    assert {"treatment_name", "pathway", "mechanism_notes"}.issubset(set(summary.columns))

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _make_initialized_trainer() -> object:
    pytest.importorskip("torch")
    from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
    from ipsc_digital_twin.models.regenai_pt_data import RegenAIPTMappings
    from ipsc_digital_twin.models.regenai_pt_trainer import RegenAIPTTrainer

    trainer = RegenAIPTTrainer(
        RegenAIPTConfig(
            input_layer="X",
            n_latent=4,
            n_hidden=8,
            n_layers=2,
            dropout=0.0,
            device="cpu",
        )
    )
    trainer.mappings = RegenAIPTMappings(
        treatment_to_id={"control": 0, "A": 1, "B": 2, "C": 3},
        id_to_treatment={0: "control", 1: "A", 2: "B", 3: "C"},
        component_to_id={"control": 0, "A": 1, "B": 2, "C": 3},
        id_to_component={0: "control", 1: "A", 2: "B", 3: "C"},
        covariate_to_id={},
        id_to_covariate={},
        gene_names=[f"g{idx}" for idx in range(6)],
        input_dim=6,
        max_components=4,
    )
    trainer.model = trainer._initialize_model(trainer.mappings)
    return trainer


def _make_adata() -> ad.AnnData:
    rng = np.random.default_rng(4)
    obs = pd.DataFrame(
        {
            "time_point": ["intermediate"] * 10 + ["post_round1"] * 10,
            "cell_type": ["iPSC"] * 10 + ["SM"] * 10,
        },
        index=[f"cell_{idx}" for idx in range(20)],
    )
    return ad.AnnData(
        X=rng.normal(size=(20, 6)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"g{idx}" for idx in range(6)]),
    )


@pytest.mark.parametrize("method", ["pca", "kernel_pca"])
def test_project_latent_space_returns_finite_coordinates(method: str) -> None:
    from ipsc_digital_twin.latent_visualization import project_latent_space

    latent = np.random.default_rng(3).normal(size=(12, 5))
    coordinates = project_latent_space(latent, method=method)

    assert coordinates.shape == (12, 2)
    assert np.isfinite(coordinates).all()


def test_extract_cell_and_drug_latent_coordinates_if_torch_available() -> None:
    from ipsc_digital_twin.latent_visualization import (
        extract_cell_latent_coordinates,
        extract_drug_latent_coordinates,
    )

    trainer = _make_initialized_trainer()
    cells = extract_cell_latent_coordinates(
        trainer,
        _make_adata(),
        color_by="cell_type",
        max_cells=12,
        batch_size=5,
    )
    drugs = extract_drug_latent_coordinates(trainer, method="kernel_pca")

    assert len(cells) == 12
    assert {"cell_id", "latent_x", "latent_y", "cell_type"}.issubset(cells.columns)
    assert drugs["component"].tolist() == ["control", "A", "B", "C"]
    control = drugs.loc[drugs["is_control"]].iloc[0]
    assert control["latent_x"] == pytest.approx(0.0)
    assert control["latent_y"] == pytest.approx(0.0)


def test_latent_plot_functions_create_png_and_csv_if_dependencies_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from ipsc_digital_twin.latent_visualization import (
        plot_cell_latent_space,
        plot_drug_latent_space,
    )

    trainer = _make_initialized_trainer()
    cell_outputs = plot_cell_latent_space(
        trainer,
        _make_adata(),
        tmp_path / "cells.png",
        color_by="time_point",
        max_cells=16,
    )
    drug_outputs = plot_drug_latent_space(trainer, tmp_path / "drugs.png")

    for output in (*cell_outputs.values(), *drug_outputs.values()):
        path = Path(output)
        assert path.exists()
        assert path.stat().st_size > 0

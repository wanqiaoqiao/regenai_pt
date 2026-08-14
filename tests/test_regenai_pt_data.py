from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ipsc_digital_twin.models import regenai_pt_data as data_mod
from ipsc_digital_twin.models.regenai_pt_config import RegenAIPTConfig
from ipsc_digital_twin.models.regenai_pt_data import (
    CategoryEncoder,
    RegenAIPTDataset,
    build_regenai_pt_dataloaders,
)


def _make_mock_adata(n_cells: int = 30) -> ad.AnnData:
    treatments = ["control", "A", "B"]
    rows = []
    for idx in range(n_cells):
        treatment = treatments[idx % len(treatments)]
        rows.append(
            {
                "treatment": treatment,
                "dose": 0.0 if treatment == "control" else float((idx % 3) + 1),
                "iPSC_line": f"line{(idx % 2) + 1}",
                "batch": f"batch{(idx % 2) + 1}",
                "round": (idx % 2) + 1,
                "time_point": "intermediate" if treatment == "control" else "post_round1",
                "replicate": f"r{(idx % 3) + 1}",
                "sequencing_run": f"run{(idx % 2) + 1}",
            }
        )
    obs = pd.DataFrame(rows)
    x = np.random.poisson(2.0, size=(n_cells, 12)).astype(float)
    var = pd.DataFrame(index=[f"g{i}" for i in range(12)])
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers["raw_counts"] = x.copy()
    adata.layers["log_normalized"] = np.log1p(x)
    return adata


def test_category_encoder_roundtrip() -> None:
    encoder = CategoryEncoder.fit(["B", "A", "A", "control"])
    encoded = encoder.transform(["A", "control"])
    assert encoded.dtype == np.int64
    assert encoder.inverse_transform(encoded.tolist()) == ["A", "control"]


def test_missing_torch_gives_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_mod, "torch", None)
    monkeypatch.setattr(data_mod, "_TORCH_IMPORT_ERROR", ImportError("No module named torch"))

    with pytest.raises(ImportError, match="PyTorch is required for RegenAI-PT"):
        data_mod.require_torch()


def test_dataset_and_dataloader_encoding_with_torch_if_available() -> None:
    torch = pytest.importorskip("torch")
    adata = _make_mock_adata()
    config = RegenAIPTConfig(
        input_layer="raw_counts",
        batch_size=8,
        covariate_keys=("replicate", "sequencing_run"),
    )

    bundle = build_regenai_pt_dataloaders(adata, config)

    total_len = len(bundle.train_dataset) + len(bundle.val_dataset) + len(bundle.test_dataset)
    assert total_len == adata.n_obs
    assert "control" in bundle.mappings.treatment_to_id
    assert "replicate" in bundle.mappings.covariate_to_id
    assert "sequencing_run" in bundle.mappings.covariate_to_id
    assert bundle.mappings.input_dim == adata.n_vars
    assert bundle.mappings.gene_names[0] == "g0"

    batch = next(iter(bundle.train_loader))
    assert isinstance(batch["x"], torch.Tensor)
    assert isinstance(batch["treatment_id"], torch.Tensor)
    assert isinstance(batch["dose_value"], torch.Tensor)
    assert isinstance(batch["component_durations"], torch.Tensor)
    assert isinstance(batch["component_duration_mask"], torch.Tensor)
    assert isinstance(batch["covariate_ids"], dict)
    assert "replicate" in batch["covariate_ids"]
    assert "sequencing_run" in batch["covariate_ids"]
    assert batch["dose_value"].dtype.is_floating_point
    assert batch["delta_target"].shape[1] == 1
    assert not batch["delta_mask"].any()
    assert not batch["component_duration_mask"].any()


def test_regenai_pt_dataset_length_equals_number_of_selected_cells_if_available() -> None:
    pytest.importorskip("torch")
    adata = _make_mock_adata(n_cells=18)
    config = RegenAIPTConfig(covariate_keys=("replicate",))
    treatment_encoder = CategoryEncoder.fit(adata.obs[config.treatment_key].astype(str).tolist())
    covariate_encoders = {
        "replicate": CategoryEncoder.fit(adata.obs["replicate"].astype(str).tolist()),
        "batch": CategoryEncoder.fit(adata.obs["batch"].astype(str).tolist()),
        "iPSC_line": CategoryEncoder.fit(adata.obs["iPSC_line"].astype(str).tolist()),
        "round": CategoryEncoder.fit(adata.obs["round"].astype(str).tolist()),
        "time_point": CategoryEncoder.fit(adata.obs["time_point"].astype(str).tolist()),
    }
    indices = np.arange(10)
    dataset = RegenAIPTDataset(
        adata=adata,
        config=config,
        treatment_encoder=treatment_encoder,
        covariate_encoders=covariate_encoders,
        indices=indices,
    )
    assert len(dataset) == 10
    item = dataset[0]
    assert int(item["treatment_id"].item()) in dataset.treatment_ids
    assert float(item["dose_value"].item()) >= 0.0
    assert int(item["batch_id"].item()) >= 0
    assert int(item["round_id"].item()) >= 0
    assert int(item["time_id"].item()) >= 0

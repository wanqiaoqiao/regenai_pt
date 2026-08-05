from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ipsc_digital_twin.cli import main
from ipsc_digital_twin.model_registry import ModelRegistry
from ipsc_digital_twin.registry import DatasetRegistry
from ipsc_digital_twin.training import (
    _write_regenai_pt_model_card,
    train_and_register_baseline,
)


def _make_trainable_adata(path: Path) -> None:
    n = 60
    x = np.random.poisson(2.0, size=(n, 10)).astype(float)
    obs = pd.DataFrame(
        {
            "time_point": ["intermediate", "post_round1", "post_round2"] * 20,
            "round1_treatment": ["A", "B", "C"] * 20,
            "round2_treatment": ["X", "Y", "Z"] * 20,
            "iPSC_line": ["line1", "line2", "line3"] * 20,
            "replicate": ["r1", "r2", "r3"] * 20,
            "target_marker_score": np.random.rand(n),
            "stress_score": np.random.rand(n),
            "off_target_score": np.random.rand(n),
        }
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(path)


def test_model_registration_and_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"model-bytes")

    reg = ModelRegistry(tmp_path / "models.json")
    rec = reg.register_model_artifact(
        artifact_path=artifact,
        model_type="baseline",
        dataset_version="1",
        training_config={"foo": "bar"},
        metrics={"mae": 0.1},
        random_seed=7,
    )

    assert rec["artifact_hash_sha256"] == reg.compute_artifact_hash(artifact)
    assert rec["model_type"] == "baseline"
    assert len(reg.list_models()) == 1


def test_training_generates_model_card(tmp_path: Path) -> None:
    ds_path = tmp_path / "train.h5ad"
    _make_trainable_adata(ds_path)

    model_reg = ModelRegistry(tmp_path / "models.json")
    out = train_and_register_baseline(
        dataset_path=ds_path,
        dataset_id="ds_1",
        dataset_version=1,
        output_dir=tmp_path / "artifacts",
        model_registry=model_reg,
        random_seed=11,
    )

    card = Path(out["model_card_path"])
    assert card.exists()
    text = card.read_text()
    assert "Model Card" in text
    assert "Random Seed" in text


def test_cli_train_and_models_commands(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    ds_path = tmp_path / "train2.h5ad"
    _make_trainable_adata(ds_path)

    ds_reg = DatasetRegistry(registry_dir / "datasets.json")
    ds_record = ds_reg.register_dataset(
        anndata_path=ds_path,
        experiment_id="exp_cli",
        preprocessing_version="p1",
        schema_version="s1",
    )

    rc_train = main(
        [
            "train",
            "--registry-dir",
            str(registry_dir),
            "--dataset-id",
            ds_record["dataset_id"],
            "--model-type",
            "baseline",
            "--output-dir",
            str(tmp_path / "train_out"),
            "--random-seed",
            "13",
        ]
    )
    assert rc_train == 0

    rc_list = main(["models", "list", "--registry-dir", str(registry_dir)])
    assert rc_list == 0

    reg = ModelRegistry(registry_dir / "models.json")
    model_id = reg.list_models()[0]["model_id"]
    rc_desc = main(["models", "describe", "--registry-dir", str(registry_dir), model_id])
    assert rc_desc == 0


def test_regenai_pt_model_card_contains_required_sections(tmp_path: Path) -> None:
    card_path = tmp_path / "regenai_pt_model_card.md"
    model_record = {
        "model_id": "model_123",
        "model_type": "regenai_pt",
        "dataset_version": 2,
        "random_seed": 17,
        "artifact_hash_sha256": "abc123",
        "git_commit": "deadbeef",
        "training_config": {"dataset_id": "ds_card"},
    }
    dataset_metadata = {
        "dataset_id": "ds_card",
        "schema_version": "schema_v2",
        "preprocessing_version": "pre_v3",
    }
    config_payload = {
        "treatment_key": "round1_treatment",
        "dose_key": "round1_dose",
        "covariate_keys": ["iPSC_line", "batch", "round", "time_point"],
        "n_latent": 64,
        "n_hidden": 128,
        "n_layers": 2,
        "input_layer": "raw_counts",
        "reconstruction_loss": "mse",
        "adversarial_weight": 1.0,
        "covariate_adversarial_weight": 1.0,
        "perturbation_adversarial_weight": 1.0,
        "embedding_l2_weight": 0.001,
        "dose_regularization_weight": 0.001,
        "weight_decay": 1e-6,
        "max_epochs": 50,
    }
    metrics = {
        "epochs_trained": 5.0,
        "best_val_reconstruction_loss": 0.123,
        "final_val_reconstruction_loss": 0.125,
        "final_train_reconstruction_loss": 0.118,
        "treatment_leakage_accuracy": 0.41,
    }
    history = {
        "val_total_loss": [0.4, 0.3],
        "val_reconstruction_loss": [0.2, 0.125],
    }

    _write_regenai_pt_model_card(
        card_path,
        model_record,
        dataset_metadata=dataset_metadata,
        config_payload=config_payload,
        metrics=metrics,
        history=history,
        notes=["Example research note."],
    )

    text = card_path.read_text(encoding='utf-8')
    assert "## Summary" in text
    assert "- Model type: regenai_pt" in text
    assert "- Training dataset ID: ds_card" in text
    assert "- Schema version: schema_v2" in text
    assert "- Preprocessing version: pre_v3" in text
    assert "## Data Interface" in text
    assert "- Treatment key: round1_treatment" in text
    assert "- Dose key: round1_dose" in text
    assert "## Architecture Summary" in text
    assert "- n_latent: 64" in text
    assert "## Loss Weights" in text
    assert "## Training Run" in text
    assert "## Reconstruction Metrics" in text
    assert "## Adversarial Leakage Metrics" in text
    assert "treatment_leakage_accuracy" in text
    assert "## Known Limitations" in text
    assert "RegenAI-PT is a custom implementation and is not the official CPA package." in text
    assert "It is not a clinical system." in text

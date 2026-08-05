from __future__ import annotations

from pathlib import Path

from ipsc_digital_twin.cli import main
from ipsc_digital_twin.registry import DatasetRegistry, ExperimentRegistry


def test_create_and_load_experiment(tmp_path: Path) -> None:
    reg_path = tmp_path / "experiments.json"
    reg = ExperimentRegistry(reg_path)

    created = reg.create_experiment(
        {
            "name": "Exp A",
            "objective": "Test objective",
            "owner": "team",
        }
    )
    loaded = reg.load_experiment(created["experiment_id"])

    assert loaded["name"] == "Exp A"
    assert created["experiment_id"] == loaded["experiment_id"]
    assert len(reg.list_experiments()) == 1


def test_register_dataset_and_version_increment_and_hash_stable(tmp_path: Path) -> None:
    ds_json = tmp_path / "datasets.json"
    ds_reg = DatasetRegistry(ds_json)

    file_path = tmp_path / "mock.h5ad"
    file_path.write_bytes(b"abc123")

    h1 = ds_reg.compute_file_hash(file_path)
    h2 = ds_reg.compute_file_hash(file_path)
    assert h1 == h2

    r1 = ds_reg.register_dataset(
        anndata_path=file_path,
        experiment_id="exp_1",
        preprocessing_version="prep_v1",
        schema_version="schema_v1",
    )
    r2 = ds_reg.register_dataset(
        anndata_path=file_path,
        experiment_id="exp_1",
        preprocessing_version="prep_v1",
        schema_version="schema_v1",
    )

    assert r1["dataset_version"] == 1
    assert r2["dataset_version"] == 2
    assert r1["file_hash_sha256"] == h1


def test_cli_registry_commands(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    rc1 = main(
        [
            "registry",
            "create-experiment",
            "--registry-dir",
            str(registry_dir),
            "--name",
            "ExpCLI",
            "--objective",
            "Obj",
            "--owner",
            "Owner",
            "--experiment-id",
            "exp_cli_1",
        ]
    )
    assert rc1 == 0

    ds_file = tmp_path / "d.h5ad"
    ds_file.write_bytes(b"xyz")

    rc2 = main(
        [
            "registry",
            "register-dataset",
            "--registry-dir",
            str(registry_dir),
            "--experiment-id",
            "exp_cli_1",
            "--anndata-path",
            str(ds_file),
            "--preprocessing-version",
            "prep_1",
            "--schema-version",
            "schema_1",
        ]
    )
    assert rc2 == 0

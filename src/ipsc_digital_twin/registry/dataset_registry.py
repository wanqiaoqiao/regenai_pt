from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DatasetRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"datasets": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text())

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def compute_file_hash(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(Path(path), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _next_version(self, experiment_id: str) -> int:
        payload = self._read()
        versions = [
            int(r["dataset_version"])
            for r in payload["datasets"]
            if r["experiment_id"] == experiment_id
        ]
        return (max(versions) + 1) if versions else 1

    def register_dataset(
        self,
        anndata_path: str | Path,
        experiment_id: str,
        preprocessing_version: str,
        schema_version: str,
    ) -> dict[str, Any]:
        p = Path(anndata_path)
        if not p.exists():
            raise FileNotFoundError(f"AnnData file does not exist: {p}")

        payload = self._read()
        file_hash = self.compute_file_hash(p)
        dataset_version = self._next_version(experiment_id)

        record = {
            "dataset_id": f"ds_{uuid.uuid4().hex[:12]}",
            "experiment_id": experiment_id,
            "anndata_path": str(p.resolve()),
            "file_hash_sha256": file_hash,
            "dataset_version": dataset_version,
            "preprocessing_version": preprocessing_version,
            "schema_version": schema_version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["datasets"].append(record)
        self._write(payload)
        return record

    def list_datasets(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        payload = self._read()
        datasets = list(payload["datasets"])
        if experiment_id is None:
            return datasets
        return [r for r in datasets if r["experiment_id"] == experiment_id]

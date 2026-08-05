from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    REQUIRED_FIELDS: tuple[str, ...] = ("name", "objective", "owner")

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"experiments": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text())

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def validate_required_metadata(self, metadata: dict[str, Any]) -> None:
        missing = [f for f in self.REQUIRED_FIELDS if not metadata.get(f)]
        if missing:
            raise ValueError(f"Missing required experiment metadata: {', '.join(missing)}")

    def create_experiment(self, metadata: dict[str, Any]) -> dict[str, Any]:
        self.validate_required_metadata(metadata)

        payload = self._read()
        experiment_id = metadata.get("experiment_id") or f"exp_{uuid.uuid4().hex[:12]}"
        if any(r["experiment_id"] == experiment_id for r in payload["experiments"]):
            raise ValueError(f"Experiment ID already exists: {experiment_id}")

        record = {
            "experiment_id": experiment_id,
            "name": str(metadata["name"]),
            "objective": str(metadata["objective"]),
            "owner": str(metadata["owner"]),
            "status": str(metadata.get("status", "active")),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata.get("metadata", {}),
        }
        payload["experiments"].append(record)
        self._write(payload)
        return record

    def load_experiment(self, experiment_id: str) -> dict[str, Any]:
        payload = self._read()
        for record in payload["experiments"]:
            if record["experiment_id"] == experiment_id:
                return record
        raise KeyError(f"Experiment not found: {experiment_id}")

    def list_experiments(self) -> list[dict[str, Any]]:
        payload = self._read()
        return list(payload["experiments"])

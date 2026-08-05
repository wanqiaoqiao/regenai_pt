from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write({"models": []})

    def _read(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text())

    def _write(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def compute_artifact_hash(artifact_path: str | Path) -> str:
        h = hashlib.sha256()
        with open(Path(artifact_path), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def get_git_commit() -> str | None:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            return out.strip() or None
        except Exception:
            return None

    def register_model_artifact(
        self,
        artifact_path: str | Path,
        model_type: str,
        dataset_version: int | str,
        training_config: dict[str, Any],
        metrics: dict[str, Any],
        random_seed: int,
    ) -> dict[str, Any]:
        p = Path(artifact_path)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found: {p}")

        payload = self._read()
        record = {
            "model_id": f"mdl_{uuid.uuid4().hex[:12]}",
            "artifact_path": str(p.resolve()),
            "artifact_hash_sha256": self.compute_artifact_hash(p),
            "model_type": model_type,
            "dataset_version": str(dataset_version),
            "training_config": training_config,
            "metrics": metrics,
            "random_seed": int(random_seed),
            "git_commit": self.get_git_commit(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["models"].append(record)
        self._write(payload)
        return record

    def list_models(self) -> list[dict[str, Any]]:
        return list(self._read()["models"])

    def describe_model(self, model_id: str) -> dict[str, Any]:
        for rec in self._read()["models"]:
            if rec["model_id"] == model_id:
                return rec
        raise KeyError(f"Model not found: {model_id}")

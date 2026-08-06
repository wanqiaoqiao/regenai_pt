from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd

ProjectionMethod = Literal["pca", "kernel_pca"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Latent-space plotting requires matplotlib. Install with "
            "`pip install -e '.[visualization]'`."
        ) from exc
    return plt


def project_latent_space(
    latent: Any,
    *,
    method: ProjectionMethod = "pca",
    gamma: float | None = None,
) -> np.ndarray:
    """Project a latent matrix to two dimensions using PCA or RBF kernel PCA."""
    matrix = np.asarray(latent, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("latent must be a two-dimensional matrix")
    if matrix.shape[0] < 2:
        raise ValueError("At least two latent observations are required")
    if not np.isfinite(matrix).all():
        raise ValueError("latent contains non-finite values")

    if method == "pca":
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = left[:, :2] * singular_values[:2]
    elif method == "kernel_pca":
        effective_gamma = float(gamma) if gamma is not None else 1.0 / max(matrix.shape[1], 1)
        if effective_gamma <= 0.0:
            raise ValueError("gamma must be positive")
        squared_norms = np.sum(np.square(matrix), axis=1, keepdims=True)
        squared_distances = np.maximum(
            squared_norms + squared_norms.T - 2.0 * matrix @ matrix.T,
            0.0,
        )
        kernel = np.exp(-effective_gamma * squared_distances)
        n_samples = kernel.shape[0]
        centering = np.eye(n_samples) - np.full((n_samples, n_samples), 1.0 / n_samples)
        centered_kernel = centering @ kernel @ centering
        eigenvalues, eigenvectors = np.linalg.eigh(centered_kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order[:2]], 0.0)
        coordinates = eigenvectors[:, order[:2]] * np.sqrt(eigenvalues)
    else:
        raise ValueError("method must be 'pca' or 'kernel_pca'")

    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    return coordinates.astype(np.float32)


def _resolve_trainer(model_or_path: Any, round_number: int) -> Any:
    if round_number not in {1, 2}:
        raise ValueError("round_number must be 1 or 2")

    from .models.regenai_pt_adapter import RegenAIPTForwardAdapter
    from .models.regenai_pt_trainer import RegenAIPTTrainer

    if isinstance(model_or_path, RegenAIPTTrainer):
        return model_or_path
    if isinstance(model_or_path, RegenAIPTForwardAdapter):
        trainer = model_or_path.round1_trainer if round_number == 1 else model_or_path.round2_trainer
        if trainer is None:
            raise ValueError(f"Combined artifact has no round-{round_number} trainer")
        return trainer

    path = Path(model_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    try:
        adapter = RegenAIPTForwardAdapter.load(path)
        trainer = adapter.round1_trainer if round_number == 1 else adapter.round2_trainer
        if trainer is not None:
            return trainer
    except (KeyError, TypeError, ValueError, RuntimeError):
        pass
    return RegenAIPTTrainer.load(path)


def _align_genes(adata: ad.AnnData, trainer: Any) -> ad.AnnData:
    if trainer.mappings is None:
        raise RuntimeError("Trainer mappings are unavailable")
    expected = list(trainer.mappings.gene_names)
    observed = adata.var_names.astype(str).tolist()
    if observed == expected:
        return adata
    missing = sorted(set(expected) - set(observed))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"AnnData is missing {len(missing)} model genes; first missing genes: {preview}")
    return adata[:, expected].copy()


def _round_relevant_cells(adata: ad.AnnData, round_number: int) -> ad.AnnData:
    if "time_point" in adata.obs.columns:
        allowed = {"intermediate", "post_round1"} if round_number == 1 else {"post_round1", "post_round2"}
        mask = adata.obs["time_point"].astype(str).isin(allowed).to_numpy()
    elif "round" in adata.obs.columns:
        allowed_rounds = {0, 1} if round_number == 1 else {1, 2}
        mask = adata.obs["round"].astype(int).isin(allowed_rounds).to_numpy()
    else:
        return adata
    if not np.any(mask):
        raise ValueError(f"AnnData contains no cells relevant to round {round_number}")
    return adata[mask].copy()


def extract_cell_latent_coordinates(
    model_or_path: Any,
    adata: ad.AnnData,
    *,
    round_number: int = 1,
    method: ProjectionMethod = "pca",
    color_by: str = "time_point",
    max_cells: int = 3000,
    batch_size: int = 256,
    random_state: int = 0,
    gamma: float | None = None,
) -> pd.DataFrame:
    """Encode sampled cells to z_basal and return auditable 2D coordinates."""
    if adata.n_obs < 2:
        raise ValueError("At least two cells are required")
    if color_by not in adata.obs.columns:
        raise KeyError(f"adata.obs does not contain color column {color_by!r}")
    if max_cells <= 1 or batch_size <= 0:
        raise ValueError("max_cells must exceed one and batch_size must be positive")

    trainer = _resolve_trainer(model_or_path, round_number)
    aligned = _round_relevant_cells(_align_genes(adata, trainer), round_number)
    rng = np.random.default_rng(random_state)
    n_selected = min(max_cells, aligned.n_obs)
    selected = np.sort(rng.choice(aligned.n_obs, size=n_selected, replace=False))

    latent_batches: list[np.ndarray] = []
    for start in range(0, n_selected, batch_size):
        batch_indices = selected[start : start + batch_size]
        latent_batches.append(trainer.encode_adata(aligned[batch_indices].copy()))
    latent = np.vstack(latent_batches)
    coordinates = project_latent_space(latent, method=method, gamma=gamma)

    frame = aligned.obs.iloc[selected].copy().reset_index(names="cell_id")
    frame["latent_x"] = coordinates[:, 0]
    frame["latent_y"] = coordinates[:, 1]
    frame["round_number"] = round_number
    frame["projection_method"] = method
    return frame


def extract_drug_latent_coordinates(
    model_or_path: Any,
    *,
    round_number: int = 1,
    method: ProjectionMethod = "kernel_pca",
    gamma: float | None = None,
    center_on_control: bool = True,
) -> pd.DataFrame:
    """Project learned component embeddings into a two-dimensional drug space."""
    trainer = _resolve_trainer(model_or_path, round_number)
    if trainer.model is None or trainer.mappings is None:
        raise RuntimeError("Trainer model or mappings are unavailable")
    embeddings = trainer.model.get_treatment_embeddings().detach().cpu().numpy()
    labels = [trainer.mappings.id_to_component[idx] for idx in range(embeddings.shape[0])]
    coordinates = project_latent_space(embeddings, method=method, gamma=gamma)

    control = trainer.config.control_treatment
    if center_on_control and control in labels:
        control_index = labels.index(control)
        coordinates = coordinates - coordinates[control_index]
    return pd.DataFrame(
        {
            "component": labels,
            "latent_x": coordinates[:, 0],
            "latent_y": coordinates[:, 1],
            "is_control": [label == control for label in labels],
            "round_number": round_number,
            "projection_method": method,
        }
    )


def plot_cell_latent_space(
    model_or_path: Any,
    adata: ad.AnnData,
    output_path: str | Path,
    *,
    round_number: int = 1,
    method: ProjectionMethod = "pca",
    color_by: str = "time_point",
    max_cells: int = 3000,
    batch_size: int = 256,
    random_state: int = 0,
    gamma: float | None = None,
) -> dict[str, str]:
    """Save a cell basal-latent scatter plot and its coordinate table."""
    plt = _require_matplotlib()
    frame = extract_cell_latent_coordinates(
        model_or_path,
        adata,
        round_number=round_number,
        method=method,
        color_by=color_by,
        max_cells=max_cells,
        batch_size=batch_size,
        random_state=random_state,
        gamma=gamma,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_path = destination.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)

    fig, axis = plt.subplots(figsize=(8.2, 6.6))
    categories = frame[color_by].astype(str)
    palette = plt.get_cmap("Dark2")
    for index, category in enumerate(sorted(categories.unique())):
        mask = categories == category
        axis.scatter(
            frame.loc[mask, "latent_x"],
            frame.loc[mask, "latent_y"],
            s=14,
            alpha=0.58,
            linewidths=0,
            color=palette(index % palette.N),
            label=category,
        )
    prefix = "kernel PC" if method == "kernel_pca" else "PC"
    axis.set_xlabel(f"{prefix}1")
    axis.set_ylabel(f"{prefix}2")
    axis.set_title(f"Round {round_number} basal cell-state latent space")
    axis.legend(title=color_by, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"plot": str(destination), "coordinates": str(csv_path)}


def plot_drug_latent_space(
    model_or_path: Any,
    output_path: str | Path,
    *,
    round_number: int = 1,
    method: ProjectionMethod = "kernel_pca",
    gamma: float | None = None,
    center_on_control: bool = True,
) -> dict[str, str]:
    """Save a labeled component-embedding plot with control-centered vectors."""
    plt = _require_matplotlib()
    frame = extract_drug_latent_coordinates(
        model_or_path,
        round_number=round_number,
        method=method,
        gamma=gamma,
        center_on_control=center_on_control,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_path = destination.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)

    fig, axis = plt.subplots(figsize=(7.2, 6.5))
    control_rows = frame[frame["is_control"]]
    origin_x = float(control_rows.iloc[0]["latent_x"]) if not control_rows.empty else 0.0
    origin_y = float(control_rows.iloc[0]["latent_y"]) if not control_rows.empty else 0.0
    palette = plt.get_cmap("Set1")
    for index, row in frame.reset_index(drop=True).iterrows():
        color = "black" if bool(row["is_control"]) else palette(index % palette.N)
        if not bool(row["is_control"]):
            axis.plot(
                [origin_x, float(row["latent_x"])],
                [origin_y, float(row["latent_y"])],
                color=color,
                alpha=0.45,
                linewidth=1.4,
            )
        axis.scatter(row["latent_x"], row["latent_y"], color=color, s=44, zorder=3)
        axis.annotate(
            str(row["component"]),
            (row["latent_x"], row["latent_y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    prefix = "kernel PC" if method == "kernel_pca" else "PC"
    axis.set_xlabel(f"{prefix}1")
    axis.set_ylabel(f"{prefix}2")
    axis.set_title(f"Round {round_number} treatment-component latent space")
    axis.axhline(0, color="#d8d8d8", linewidth=0.8, zorder=0)
    axis.axvline(0, color="#d8d8d8", linewidth=0.8, zorder=0)
    axis.grid(alpha=0.1)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"plot": str(destination), "coordinates": str(csv_path)}


__all__ = [
    "extract_cell_latent_coordinates",
    "extract_drug_latent_coordinates",
    "plot_cell_latent_space",
    "plot_drug_latent_space",
    "project_latent_space",
]

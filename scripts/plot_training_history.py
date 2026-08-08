from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

from ipsc_digital_twin.latent_visualization import plot_cell_latent_space

INPUT_DIR = Path("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/outputs/regenai_pt_v0.7.0_combined_lr1e-3_epoch_20")
ADATA_PATH = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/"
    "sample_data/prepared_h5ad_2026_07/regenai_pt_train_only.h5ad"
)
OUTPUT_DIR = INPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LATENT_CELLS = 3000

history_path = next(
    INPUT_DIR.glob("*_training_history.csv"),
    None,
)

if history_path is None:
    raise FileNotFoundError(
        f"No training-history CSV found in {INPUT_DIR}"
    )

history = pd.read_csv(history_path)

print("History:", history_path)
print("Columns:", history.columns.tolist())


def plot_metric(
    frame: pd.DataFrame,
    train_column: str,
    validation_column: str,
    title: str,
    filename: str,
    y_label: str = "Loss",
) -> None:
    if train_column not in frame.columns:
        print(f"Skipping missing column: {train_column}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    phases = (
        frame["phase"].dropna().unique()
        if "phase" in frame.columns
        else ["model"]
    )

    for phase in phases:
        if "phase" in frame.columns:
            subset = frame.loc[frame["phase"] == phase].copy()
        else:
            subset = frame.copy()

        subset = subset.sort_values("epoch")

        ax.plot(
            subset["epoch"],
            subset[train_column],
            marker="o",
            linewidth=2,
            label=f"{phase} train",
        )

        if validation_column in subset.columns:
            ax.plot(
                subset["epoch"],
                subset[validation_column],
                marker="s",
                linestyle="--",
                linewidth=2,
                label=f"{phase} validation",
            )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


plot_metric(
    history,
    "train_total_loss",
    "val_total_loss",
    "Total training loss",
    "total_loss.png",
)

plot_metric(
    history,
    "train_reconstruction_loss",
    "val_reconstruction_loss",
    "Expression reconstruction loss",
    "reconstruction_loss.png",
)

plot_metric(
    history,
    "train_treatment_adv_loss",
    "val_treatment_adv_loss",
    "Treatment adversarial loss",
    "treatment_adversarial_loss.png",
)

plot_metric(
    history,
    "train_covariate_adv_loss",
    "val_covariate_adv_loss",
    "Covariate adversarial loss",
    "covariate_adversarial_loss.png",
)
plot_metric(
    history,
    "train_covariate_adv_accuracy",
    "val_covariate_adv_accuracy",
    "Covariate adversary macro accuracy",
    "covariate_adversarial_accuracy.png",
    y_label="Accuracy",
)

for train_column in sorted(
    column
    for column in history.columns
    if column.startswith("train_covariate_adv_accuracy_")
):
    covariate_key = train_column.removeprefix("train_covariate_adv_accuracy_")
    safe_key = "".join(character if character.isalnum() else "_" for character in covariate_key)
    plot_metric(
        history,
        train_column,
        f"val_covariate_adv_accuracy_{covariate_key}",
        f"Covariate adversary accuracy: {covariate_key}",
        f"covariate_adversarial_accuracy_{safe_key}.png",
        y_label="Accuracy",
    )

plot_metric(
    history,
    "train_embedding_l2_loss",
    "val_embedding_l2_loss",
    "Embedding regularization loss",
    "embedding_regularization.png",
)

plot_metric(
    history,
    "train_dose_regularization_loss",
    "val_dose_regularization_loss",
    "Dose regularization loss",
    "dose_regularization.png",
)
plot_metric(
    history,
    "train_mean",
    "val_mean",
    "mean",
    "mean.png",
)
plot_metric(
    history,
    "train_Var",
    "val_Var",
    "Var",
    "Var.png",
)
plot_metric(
    history,
    "train_mean_DE",
    "val_mean_DE",
    "mean_DE",
    "mean_DE.png",
)
plot_metric(
    history,
    "train_Var_DE",
    "val_Var_DE",
    "Var_DE",
    "Var_DE.png",
)


def plot_cell_latent_by_line() -> None:
    """Plot round-specific basal latent spaces colored by iPSC cell line."""
    model_path = next(INPUT_DIR.glob("*_regenai_pt_combined_model.pt"), None)
    if model_path is None:
        raise FileNotFoundError(f"No combined RegenAI-PT model found in {INPUT_DIR}")
    if not ADATA_PATH.exists():
        raise FileNotFoundError(f"Training AnnData not found: {ADATA_PATH}")

    adata = ad.read_h5ad(ADATA_PATH)
    if "iPSC_line" not in adata.obs.columns:
        raise KeyError("Training AnnData does not contain adata.obs['iPSC_line']")

    line_counts = adata.obs["iPSC_line"].astype(str).value_counts()
    print("iPSC lines:", line_counts.to_dict())
    if len(line_counts) < 2:
        print("Warning: only one iPSC line is available; the plot will have one color.")

    for round_number in (1, 2):
        result = plot_cell_latent_space(
            model_path,
            adata,
            OUTPUT_DIR / f"cell_latent_round{round_number}_by_iPSC_line.png",
            round_number=round_number,
            color_by="iPSC_line",
            method="pca",
            max_cells=MAX_LATENT_CELLS,
            batch_size=256,
            random_state=0,
        )
        print(f"Saved round {round_number} cell-line latent plot:", result["plot"])
        print(f"Saved round {round_number} latent coordinates:", result["coordinates"])


plot_cell_latent_by_line()

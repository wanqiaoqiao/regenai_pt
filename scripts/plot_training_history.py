from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = Path("outputs/regenai_pt_combined")
OUTPUT_DIR = INPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    ax.set_ylabel("Loss")
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
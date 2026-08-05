from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


COMBINED_PATH = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/prepared_h5ad_2026_07/combined_regenai_pt_training_input.h5ad"
)
TRAIN_PATH = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/prepared_h5ad_2026_07/regenai_pt_train_only.h5ad"
)
HELDOUT_PATH = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/prepared_h5ad_2026_07/GSM4876138_heldout_test.h5ad"
)


def add_log_normalized_layer(adata: ad.AnnData) -> None:
    """Create log1p(CP10K) expression without changing raw counts."""
    source = adata.layers.get("raw_counts", adata.X)

    if sparse.issparse(source):
        matrix = source.astype(np.float32).tocsr(copy=True)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(
            10_000.0,
            totals,
            out=np.zeros_like(totals, dtype=np.float32),
            where=totals > 0,
        )
        matrix = sparse.diags(scale) @ matrix
        matrix = matrix.tocsr()
        matrix.data = np.log1p(matrix.data)
    else:
        matrix = np.asarray(source, dtype=np.float32).copy()
        totals = matrix.sum(axis=1)
        scale = np.divide(
            10_000.0,
            totals,
            out=np.zeros_like(totals, dtype=np.float32),
            where=totals > 0,
        )
        matrix *= scale[:, None]
        matrix = np.log1p(matrix)

    adata.layers["log_normalized"] = matrix


combined = ad.read_h5ad(COMBINED_PATH)

test_mask = (
    combined.obs["sample_id"].astype(str) == "GSM4876138"
)

train = combined[~test_mask].copy()
heldout = combined[test_mask].copy()

if train.n_obs == 0:
    raise ValueError("Training subset is empty")

if heldout.n_obs == 0:
    raise ValueError("No GSM4876138 cells were found")

if "GSM4876138" in set(train.obs["sample_id"].astype(str)):
    raise RuntimeError("Held-out cells leaked into training data")

add_log_normalized_layer(train)
add_log_normalized_layer(heldout)

TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
train.write_h5ad(TRAIN_PATH)
heldout.write_h5ad(HELDOUT_PATH)

print("Training:", train)
print("Held out:", heldout)
print("Training samples:")
print(train.obs["sample_id"].value_counts())
print("Held-out samples:")
print(heldout.obs["sample_id"].value_counts())
print("Training layers:", list(train.layers))
print("Saved:", TRAIN_PATH.resolve())
print("Saved:", HELDOUT_PATH.resolve())
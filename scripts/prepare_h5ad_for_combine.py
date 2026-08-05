from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np


DATA_DIR = Path("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/h5ad_2026_07")
OUTPUT_PATH = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/prepared_h5ad_2026_07/combined_regenai_pt_training_input.h5ad"
)

HELD_OUT_SAMPLE = "GSM4876138"
RANDOM_SEED = 42

# Replace treatment placeholders with exact components, doses, and units
# from the experimental protocols.
SAMPLES = [
    {
        "sample_id": "GSM7147764",
        "filename": "GSM7147764_raw.h5ad",
        "iPSC_line": "CS0007iCTR-n5",
        "cell_type_observed": "iPSC",
        "day": 0,
        "round": 0,
        "time_point": "intermediate",
        "round1_components": ["control"],
        "round1_doses": [0.0],
        "round1_dose_units": ["none"],
        "round2_components": ["control"],
        "round2_doses": [0.0],
        "round2_dose_units": ["none"],
    },
    {
        "sample_id": "GSM7147766",
        "filename": "GSM7147766_raw.h5ad",
        "iPSC_line": "CS0007iCTR-n5",
        "cell_type_observed": "SM",
        "day": 8,
        "round": 1,
        "time_point": "post_round1",
        "round1_components": ["A"],
        "round1_doses": [5.0],
        "round1_dose_units": ["uM"],
        "round2_components": ["control"],
        "round2_doses": [0.0],
        "round2_dose_units": ["none"],
    },
    {
        "sample_id": "GSM7147768",
        "filename": "GSM7147768_raw.h5ad",
        "iPSC_line": "CS0007iCTR-n5",
        "cell_type_observed": "SM",
        "day": 32,
        "round": 2,
        "time_point": "post_round2",
        "round1_components": ["A"],
        "round1_doses": [5.0],
        "round1_dose_units": ["uM"],
        "round2_components": ["B"],
        "round2_doses": [5.0],
        "round2_dose_units": ["uM"],
    },
    {
        "sample_id": "GSM7147769",
        "filename": "GSM7147769_raw.h5ad",
        "iPSC_line": "CS0007iCTR-n5",
        "cell_type_observed": "SM",
        "day": 32,
        "round": 2,
        "time_point": "post_round2",
        "round1_components": ["A"],
        "round1_doses": [5.0],
        "round1_dose_units": ["uM"],
        "round2_components": ["B","C59"],
        "round2_doses": [5.0,5.0],
        "round2_dose_units": ["uM","uM"],
    },
    {
        "sample_id": "GSM4876130",
        "filename": "GSM4876130_raw.h5ad",
        "iPSC_line": "STAN061i-164-1",
        "cell_type_observed": "Cp",
        "day": 0,
        "round": 0,
        "time_point": "intermediate",
        "round1_components": ["control"],
        "round1_doses": [0.0],
        "round1_dose_units": ["none"],
        "round2_components": ["control"],
        "round2_doses": [0.0],
        "round2_dose_units": ["none"],
    },
    {
        "sample_id": "GSM4876132",
        "filename": "GSM4876132_raw.h5ad",
        "iPSC_line": "STAN061i-164-1",
        "cell_type_observed": "Cp-D7",
        "day": 7,
        "round": 1,
        "time_point": "post_round1",
        "round1_components": ["C"],
        "round1_doses": [5.0],
        "round1_dose_units": ["uM"],
        "round2_components": ["control"],
        "round2_doses": [0.0],
        "round2_dose_units": ["none"],
    },
    {
        "sample_id": "GSM4876134",
        "filename": "GSM4876134_raw.h5ad",
        "iPSC_line": "STAN061i-164-1",
        "cell_type_observed": "Cp-D28",
        "day": 28,
        "round": 2,
        "time_point": "post_round2",
        "round1_components": ["C"],
        "round1_doses": [5.0],
        "round1_dose_units": ["uM"],
        "round2_components": ["D"],
        "round2_doses": [5.0],
        "round2_dose_units": ["uM"],
    },
    {
        "sample_id": "GSM4876136",
        "filename": "GSM4876136_raw.h5ad",
        "iPSC_line": "STAN061i-164-1",
        "cell_type_observed": "Cp-D7",
        "day": 7,
        "round": 1,
        "time_point": "post_round1",
        "round1_components": ["C","C59"],
        "round1_doses": [5.0, 5.0],
        "round1_dose_units": ["uM","uM"],
        "round2_components": ["control"],
        "round2_doses": [0.0],
        "round2_dose_units": ["none"],
    },
    {
        "sample_id": "GSM4876138",
        "filename": "GSM4876138_raw.h5ad",
        "iPSC_line": "STAN061i-164-1",
        "cell_type_observed": "Cp-D28",
        "day": 28,
        "round": 2,
        "time_point": "post_round2",
        "round1_components": ["C","C59"],
        "round1_doses": [5.0,5.0],
        "round1_dose_units": ["uM","uM"],
        "round2_components": ["D","C59"],
        "round2_doses": [5.0,5.0],
        "round2_dose_units": ["uM","uM"],
    },
    # Add the remaining samples using the metadata table above.
]


def as_json(values: list[str] | list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


def annotate_sample(adata: ad.AnnData, metadata: dict[str, object]) -> ad.AnnData:
    adata = adata.copy()
    sample_id = str(metadata["sample_id"])

    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names = [
        f"{sample_id}:{barcode}" for barcode in adata.obs_names.astype(str)
    ]

    for key in (
        "sample_id",
        "iPSC_line",
        "cell_type_observed",
        "day",
        "round",
        "time_point",
    ):
        adata.obs[key] = metadata[key]

    for round_number in (1, 2):
        components = metadata[f"round{round_number}_components"]
        doses = metadata[f"round{round_number}_doses"]
        units = metadata[f"round{round_number}_dose_units"]

        if not (len(components) == len(doses) == len(units)):
            raise ValueError(
                f"{sample_id}: round {round_number} component, dose, "
                "and unit lengths differ"
            )

        adata.obs[f"round{round_number}_components"] = as_json(components)
        adata.obs[f"round{round_number}_doses"] = as_json(doses)
        adata.obs[f"round{round_number}_dose_units"] = as_json(units)
        adata.obs[f"round{round_number}_treatment"] = "+".join(components)
        adata.obs[f"round{round_number}_dose"] = float(sum(doses))

    r1 = "+".join(metadata["round1_components"])
    r2 = "+".join(metadata["round2_components"])
    adata.obs["treatment_sequence"] = f"{r1}->{r2}"

    adata.obs["experiment_id"] = "combined_ipsc_differentiation"
    adata.obs["batch"] = sample_id
    adata.obs["sequencing_run"] = "unknown"
    adata.obs["replicate"] = sample_id

    # Preserve counts before later normalization. Confirm that X contains counts.
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()

    if sample_id == HELD_OUT_SAMPLE:
        adata.obs["data_split"] = "test"
    else:
        rng = np.random.default_rng(
            RANDOM_SEED + sum(sample_id.encode("utf-8"))
        )
        validation = rng.random(adata.n_obs) < 0.20
        adata.obs["data_split"] = np.where(
            validation, "validation", "train"
        )

    return adata


datasets: list[ad.AnnData] = []

for metadata in SAMPLES:
    input_path = DATA_DIR / str(metadata["filename"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    dataset = ad.read_h5ad(input_path)
    datasets.append(annotate_sample(dataset, metadata))

# Restrict to genes present in every dataset.
common_genes = list(datasets[0].var_names)
for dataset in datasets[1:]:
    available = set(dataset.var_names)
    common_genes = [gene for gene in common_genes if gene in available]

if not common_genes:
    raise ValueError("The datasets have no common genes")

datasets = [dataset[:, common_genes].copy() for dataset in datasets]

combined = ad.concat(
    datasets,
    join="inner",
    merge="same",
    label="source_dataset",
    keys=[str(item["sample_id"]) for item in SAMPLES],
    index_unique=None,
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
combined.write_h5ad(OUTPUT_PATH)

print(combined)
print(combined.obs["data_split"].value_counts())
print(combined.obs.groupby(["sample_id", "data_split"]).size())
print(f"Saved: {OUTPUT_PATH}")
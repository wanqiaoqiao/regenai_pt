from pathlib import Path

import anndata as ad

adata = ad.read_h5ad("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/GSM7147764_sample_1/GSM7147764_sample_1_raw.h5ad")

print(adata)
print("obs columns:", adata.obs.columns.tolist())
print("var count:", adata.n_vars)
print("obs count:", adata.n_obs)
print("layers:", list(adata.layers.keys()))
print("first genes:", adata.var_names[:10].tolist())

before = ad.read_h5ad("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/GSM7147764_sample_1/GSM7147764_sample_1_raw.h5ad")
after_r1 = ad.read_h5ad("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/GSM7147766_sample_3/GSM7147766_sample_3_raw.h5ad")
after_r2 = ad.read_h5ad("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/GSM7147769_sample_6/GSM7147769_sample_6_raw.h5ad")

# Example protocol metadata: replace with your real values
round1_treatment = "Treatment_A"
round1_dose = 3.0
round2_treatment = "Treatment_B"
round2_dose = 10.0
ipsc_line = "line_1"
batch = "batch_1"
replicate = "rep_1"
sequencing_run = "run_1"

def annotate_stage(
    adata,
    time_point,
    round_number,
):
    adata = adata.copy()
    adata.obs["time_point"] = time_point
    adata.obs["round"] = round_number
    adata.obs["round1_treatment"] = round1_treatment
    adata.obs["round1_dose"] = round1_dose
    adata.obs["round2_treatment"] = round2_treatment
    adata.obs["round2_dose"] = round2_dose
    adata.obs["treatment_sequence"] = f"{round1_treatment}->{round2_treatment}"
    adata.obs["iPSC_line"] = ipsc_line
    adata.obs["batch"] = batch
    adata.obs["replicate"] = replicate
    adata.obs["sequencing_run"] = sequencing_run
    return adata

before = annotate_stage(before, "intermediate", 0)
after_r1 = annotate_stage(after_r1, "post_round1", 1)
after_r2 = annotate_stage(after_r2, "post_round2", 2)

# Optional: keep track of source file
before.obs["source_file"] = "GSM7147764_sample_1_raw.h5ad"
after_r1.obs["source_file"] = "GSM7147766_sample_3_raw.h5ad"
after_r2.obs["source_file"] = "GSM7147769_sample_6_raw.h5ad"

# Make cell IDs unique
before.obs_names = [f"before_{x}" for x in before.obs_names]
after_r1.obs_names = [f"r1_{x}" for x in after_r1.obs_names]
after_r2.obs_names = [f"r2_{x}" for x in after_r2.obs_names]

# Concatenate on shared genes only
combined = ad.concat(
    [before, after_r1, after_r2],
    join="inner",
    axis=0,
    merge="same",
    label="dataset_stage",
    keys=["before", "after_round1", "after_round2"],
)

output_path = Path("/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/training_data_combined/combined_two_round_training_input.h5ad")
output_path.parent.mkdir(parents=True, exist_ok=True)
combined.write_h5ad(output_path)
print(f"Saved combined AnnData to: {output_path}")

print(combined)
print(combined.obs.columns.tolist())

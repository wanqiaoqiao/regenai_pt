from pathlib import Path
import scanpy as sc

data_dir = Path(
    "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/sample_data/GSM4876138"
)

prefix = "GSM4876138_C59_D28_"

print("Running script:", __file__)
print("Data directory:", data_dir)
print("Prefix:", prefix)
print("Files found:")
for file_path in sorted(data_dir.iterdir()):
    print("  ", file_path.name)

adata = sc.read_10x_mtx(
    data_dir,
    prefix=prefix,
    var_names="gene_symbols",
    make_unique=True
)

adata.obs["sample_id"] = "GSM4876138"
adata.obs["sample_name"] = "Sample_1"
adata.obs["cell_barcode"] = adata.obs_names.astype(str)

adata.layers["counts"] = adata.X.copy()

print(adata)
print(f"Number of cells: {adata.n_obs:,}")
print(f"Number of genes/features: {adata.n_vars:,}")
print(f"Matrix type: {type(adata.X)}")

output_file = data_dir / "GSM4876138_raw.h5ad"
adata.write_h5ad(output_file, compression="gzip")

print(f"Saved: {output_file}")
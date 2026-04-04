import os
import time
import psutil
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp

# -----------------------------
# Settings
# -----------------------------
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

process = psutil.Process(os.getpid())

def get_cpu_mem_mb():
    return process.memory_info().rss / 1024**2

def get_gpu_mem_mb():
    if not use_cuda:
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


# -----------------------------
# Start measurement
# -----------------------------
start_time = time.perf_counter()
cpu_mem_start = get_cpu_mem_mb()

if use_cuda:
    torch.cuda.reset_peak_memory_stats()

# -----------------------------
# Load data
# -----------------------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad')

st_data = data[data.obs['Section ID'] == '0503_F4_C',].copy()
st_data.layers['counts'] = st_data.X
st_data.obs['tech'] = 'st'

section_ids = [
    "0503_F5_T", "0503_F5_C", "0503_F5_L", "0503_F5_S", "0503_M4_S",
    "0503_F4_T", "0503_F3_C", "0503_F4_S", "0503_F3_L", "0503_F3_T",
    "0503_M5_C", "0503_F3_S", "0503_M5_T", "0503_M4_C", "0503_M5_S",
    "0503_M4_L", "0503_M4_T"
]

series_dir = '/maiziezhou_lab2/yuling/label_Transfer/scVI/0503_F4_C'
os.makedirs(series_dir, exist_ok=True)

sc_adata = data[data.obs["Section ID"].isin(section_ids)].copy()
sc_adata.layers['counts'] = sc_adata.X
sc_adata.obs['tech'] = 'sc'

adata = anndata.concat([st_data, sc_adata])
adata.layers["counts"] = adata.X.copy()

# -----------------------------
# Preprocessing
# -----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=500,
    layer="counts",
    batch_key="tech",
    subset=True,
)

# -----------------------------
# SCVI
# -----------------------------
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="tech"
)

scvi_model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_latent=30
)

scvi_model.train()

adata.obsm["X_scVI"] = scvi_model.get_latent_representation()

# -----------------------------
# SCANVI
# -----------------------------
SCANVI_CELLTYPE_KEY = "celltype_scanvi"

adata.obs[SCANVI_CELLTYPE_KEY] = "Unknown"
sc_mask = adata.obs["tech"] == "sc"
adata.obs.loc[sc_mask, SCANVI_CELLTYPE_KEY] = (
    adata.obs.loc[sc_mask, "MERFISH cell type annotation"].values
)

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=adata,
    unlabeled_category="Unknown",
    labels_key=SCANVI_CELLTYPE_KEY,
)

scanvi_model.train(
    max_epochs=20,
    n_samples_per_label=100
)

adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation(adata)
adata.obs["C_scANVI"] = scanvi_model.predict(adata)

# -----------------------------
# UMAP (optional, but included)
# -----------------------------
sc.pp.neighbors(adata, use_rep="X_scANVI")
sc.tl.umap(adata, min_dist=0.3)

# -----------------------------
# Save predictions
# -----------------------------
pred = adata[adata.obs[SCANVI_CELLTYPE_KEY] == "Unknown"]
pred.obs.to_csv(
    osp.join(series_dir, "label_transfer.csv"),
    index=True
)

# -----------------------------
# End measurement
# -----------------------------
elapsed_time = time.perf_counter() - start_time
cpu_mem_peak = get_cpu_mem_mb()
gpu_mem_peak = get_gpu_mem_mb()

# -----------------------------
# Save runtime + memory
# -----------------------------
log_df = pd.DataFrame([{
   
    "runtime_sec": elapsed_time,
    "peak_memory_MiB": gpu_mem_peak
}])

log_path = osp.join(series_dir, "runtimeSec_memoryMiB.csv")
log_df.to_csv(log_path, index=False)

print(log_df)
print("Saved runtime/memory log to:", log_path)

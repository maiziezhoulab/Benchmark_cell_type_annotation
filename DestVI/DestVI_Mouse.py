import os
import time
import psutil
import os.path as osp
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scvi.model import CondSCVI, DestVI

# -----------------------------
# Settings
# -----------------------------
scvi.settings.seed = 0
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
# Output directory
# -----------------------------
base_dir = '/maiziezhou_lab2/yuling/label_Transfer/DestVI/0503_F4_C'
os.makedirs(base_dir, exist_ok=True)

# -----------------------------
# Start timing & memory
# -----------------------------
start_time = time.perf_counter()
cpu_mem_start = get_cpu_mem_mb()

if use_cuda:
    torch.cuda.reset_peak_memory_stats()

# -----------------------------
# Load data
# -----------------------------
ad_train = sc.read_h5ad(
    '/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad'
)

section_ids = [
    "0503_F5_T", "0503_F5_C", "0503_F5_L", "0503_F5_S", "0503_M4_S",
    "0503_F4_T", "0503_F3_C", "0503_F4_S", "0503_F3_L", "0503_F3_T",
    "0503_M5_C", "0503_F3_S", "0503_M5_T", "0503_M4_C", "0503_M5_S",
    "0503_M4_L", "0503_M4_T"
]

# scRNA-seq reference
sc_adata = ad_train[
    ad_train.obs["Section ID"].isin(section_ids)
].copy()
sc_adata.layers["counts"] = sc_adata.X

# spatial query
adata = sc.read_h5ad(
    '/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad'
)
st_data = adata[
    adata.obs["Section ID"] == "0503_F4_C"
].copy()
st_data.layers["counts"] = st_data.X

# -----------------------------
# CondSCVI (reference model)
# -----------------------------
CondSCVI.setup_anndata(
    sc_adata,
    layer="counts",
    labels_key="MERFISH cell type annotation"
)

sc_model = CondSCVI(
    sc_adata,
    weight_obs=False
)

sc_model.view_anndata_setup()
sc_model.train()

# -----------------------------
# DestVI (spatial model)
# -----------------------------
DestVI.setup_anndata(
    st_data,
    layer="counts"
)

st_model = DestVI.from_rna_model(
    st_data,
    sc_model
)

st_model.view_anndata_setup()
st_model.train(max_epochs=2500)

# -----------------------------
# Save proportions
# -----------------------------
st_data.obsm["proportions"] = st_model.get_proportions()

df = st_data.obsm["proportions"].copy()
df.to_csv(
    osp.join(base_dir, "proportions.csv"),
    index=True
)

# -----------------------------
# End timing & memory
# -----------------------------
elapsed_time = time.perf_counter() - start_time
cpu_mem_peak = get_cpu_mem_mb()
gpu_mem_peak = get_gpu_mem_mb()

# -----------------------------
# Save runtime & memory log
# -----------------------------
log_df = pd.DataFrame([{
   
    "runtime_sec": elapsed_time,
    "peak_memory_MiB": gpu_mem_peak
}])

log_path = osp.join(base_dir, "runtimeSec_memoryMiB.csv")
log_df.to_csv(log_path, index=False)

print(log_df)
print("Saved runtime/memory log to:", log_path)

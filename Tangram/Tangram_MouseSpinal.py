import os
import time
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
import torch

# -----------------------------
# Settings: conda activate loki_env
# -----------------------------
use_cuda = False            # set True if you want GPU
device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

process = psutil.Process(os.getpid())

def get_cpu_mem_mb():
    return process.memory_info().rss / 1024**2

def get_gpu_mem_mb():
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


# -----------------------------
# Load data
# -----------------------------
rna_path = "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
rna = sc.read_h5ad(rna_path)

# Select sections
unique_section = rna.obs["Section ID"].unique()
selected_0503 = [s for s in unique_section if s.startswith("0503")]
selected_0503_clean = [s for s in selected_0503 if s != "0503_nan_nan"]
selected_0503_1 = [s for s in selected_0503_clean if s != "0503_F4_C"]

# Output directory
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/0503_F4_C_output"
os.makedirs(outdir, exist_ok=True)

# Query (ST)
query_data = rna[rna.obs["Section ID"] == "0503_F4_C"].copy()

# Reference (scRNA)
reference_data = rna[rna.obs["Section ID"].isin(selected_0503_1)].copy()

# -----------------------------
# Preprocessing
# -----------------------------
sc.pp.normalize_total(query_data)
sc.pp.log1p(query_data)

sc.pp.normalize_total(reference_data)
sc.pp.log1p(reference_data)

# Harmonize genes
tg.pp_adatas(reference_data, query_data, genes=None)
# -----------------------------
# Start timing & memory
# -----------------------------
start_time = time.perf_counter()
cpu_mem_start = get_cpu_mem_mb()

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
# -----------------------------
# Tangram mapping
# -----------------------------
tg_map = tg.map_cells_to_space(
    reference_data,
    query_data,
    density_prior="uniform",
    device=device
)

# Project cell type annotation
tg.project_cell_annotations(
    adata_sp=query_data,
    adata_map=tg_map,
    annotation="MERFISH cell type annotation"
)
# ==============================
# END benchmark region
# ==============================
elapsed_time = time.perf_counter() - start_time
cpu_mem_peak = get_cpu_mem_mb()
gpu_mem_peak = get_gpu_mem_mb()
# -----------------------------
# Save predictions
# -----------------------------
df = pd.DataFrame(
    query_data.obsm["tangram_ct_pred"],
    index=query_data.obs_names
)

outfile = os.path.join(outdir, "tangram_ct_pred_17_slices.csv")
df.to_csv(outfile)


# -----------------------------
# Save runtime & memory log
# -----------------------------
log_df = pd.DataFrame([{
   
    "runtime_sec": elapsed_time,
    "peak_memory_MiB": gpu_mem_peak
}])

log_path = os.path.join(outdir, "runtimeSec_memoryMiB.csv")
log_df.to_csv(log_path, index=False)

print(log_df)
print("Saved Tangram output to:", outfile)
print("Saved runtime/memory log to:", log_path)

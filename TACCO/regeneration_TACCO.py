############################################
# TACCO benchmark: running time + peak memory
############################################

import os
import time
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tacco as tc

############################################
# Helper: convert AnnData to float32
############################################
def make_float32(adata):
    if adata.is_view:
        adata = adata.copy()

    if getattr(adata, "isbacked", False):
        import anndata as ad
        adata = ad.read_h5ad(adata.filename, backed=None)

    if sp.issparse(adata.X):
        adata.X = adata.X.asfptype().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata


############################################
# Paths
############################################
h5ad_path = "/maiziezhou_lab2/yuling/datasets/Develop/5DPIs.h5ad"
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/tacco/regeneration_output"
os.makedirs(outdir, exist_ok=True)

############################################
# Load data (NOT benchmarked)
############################################
data = sc.read_h5ad(h5ad_path)

reference_data = data[
   data.obs['Batch'].isin([
       'Injury_5DPI_rep1_SS200000147BL_D2',
     'Injury_5DPI_rep2_SS200000147BL_D2'
  ]),
].copy()
#stage_54.obs['Annotation'] = stage_54.obs['Annotation'].astype('category')
new_names = (
    reference_data.obs_names.astype(str)
    + "_"
    + reference_data.obs["Batch"].astype(str)
)

new_names = pd.Index(new_names).str.replace(r"\s+", "_", regex=True)
if new_names.has_duplicates:
    new_names = pd.Index(pd.io.parsers.ParserBase({"names": new_names})._maybe_dedup_names(new_names))
reference_data.obs_names = new_names
puck = data[
    data.obs["Batch"] == "Injury_5DPI_rep3_SS200000147BL_D3"
].copy()

############################################
# Prepare data (NOT benchmarked)
############################################
reference_data = make_float32(reference_data)
puck = make_float32(puck)

if "counts" in puck.layers:
    L = puck.layers["counts"]
    puck.X = (
        L.asfptype().astype(np.float32)
        if sp.issparse(L)
        else np.asarray(L, dtype=np.float32)
    )

if "counts" in reference_data.layers:
    L = reference_data.layers["counts"]
    reference_data.X = (
        L.asfptype().astype(np.float32)
        if sp.issparse(L)
        else np.asarray(L, dtype=np.float32)
    )

############################################
# Benchmark START
############################################
process = psutil.Process(os.getpid())

mem_before = process.memory_info().rss  # bytes
t_start = time.perf_counter()

tc.tl.annotate(
    puck,
    reference_data,
    annotation_key="Annotation",
    result_key="ClusterName",
    counts_location="X",
    assume_valid_counts=True,
)

t_end = time.perf_counter()
mem_after = process.memory_info().rss
############################################
# Benchmark END
############################################

############################################
# Metrics
############################################
runtime_sec = t_end - t_start
peak_mem_MiB = max(mem_before, mem_after) / 1024**2

############################################
# Save benchmark result
############################################
summary_df = pd.DataFrame(
    {
        "Time_sec": [runtime_sec],
        "Peak_Memory": [peak_mem_MiB],
    }
)

summary_df.to_csv(
    os.path.join(outdir, "runtimeSec_memoryMiB.csv"),
    index=False,
)

############################################
# Save prediction (optional)
############################################
pred = puck.obsm["ClusterName"]
pred.to_csv(os.path.join(outdir, "TACCO_prediction.csv"))

############################################
# Print summary
############################################
print("TACCO benchmark finished")
print(f"Elapsed time (sec): {runtime_sec:.2f}")
print(f"Peak RAM used (MiB): {peak_mem_MiB:.2f}")
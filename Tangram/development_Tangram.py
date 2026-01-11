############################################
# Tangram benchmark: running time + peak memory
############################################

import os
import time
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
import torch

############################################
# Paths
############################################
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/Development_output"
os.makedirs(outdir, exist_ok=True)

h5ad_path = "/maiziezhou_lab2/yuling/datasets/Development.h5ad"

############################################
# Load data (NOT benchmarked)
############################################
data = sc.read_h5ad(h5ad_path)

reference_data = data[
    data.obs["Batch"] == "Stage54_telencephalon_rep2_DP8400015649BRD6_2"
].copy()

query_data = data[
    data.obs["Batch"] == "Stage44_telencephalon_rep2_FP200000239BL_E4"
].copy()

############################################
# Preprocessing (NOT benchmarked)
############################################
reference = reference_data.copy()
query = query_data.copy()

sc.pp.normalize_total(reference)
sc.pp.log1p(reference)

sc.pp.normalize_total(query)
sc.pp.log1p(query)

sc.pp.highly_variable_genes(
    reference,
    n_top_genes = 3000,
    flavor="seurat_v3"
)

hvg = reference.var_names[reference.var["highly_variable"]].to_list()
genes = list(set(hvg).intersection(set(query.var_names)))

reference = reference[:, genes].copy()
query = query[:, genes].copy()

tg.pp_adatas(reference, query, genes=None)

############################################
# Benchmark START
############################################
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss  # bytes

t_start = time.perf_counter()

tg_map = tg.map_cells_to_space(
    reference,
    query,
    density_prior="uniform",
    device="cuda"  # change to "cpu" if no GPU
)

tg.project_cell_annotations(
    adata_sp=query,
    adata_map=tg_map,
    annotation="Annotation"
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
        "Elapsed_Time_sec": [runtime_sec],
        "Peak_RAM_Used_MiB": [peak_mem_MiB],
    }
)

summary_df.to_csv(
    os.path.join(outdir, "runtimeSec_memoryMiB.csv"),
    index=False,
)

############################################
# Save predictions (optional)
############################################
pred_df = pd.DataFrame(
    query.obsm["tangram_ct_pred"],
    index=query.obs_names
)

pred_df.to_csv(
    os.path.join(outdir, "tangram_prediction_1.csv")
)

############################################
# Print summary
############################################
print("Tangram benchmark finished")
print(f"Elapsed time (sec): {runtime_sec:.2f}")
print(f"Peak RAM used (MiB): {peak_mem_MiB:.2f}")

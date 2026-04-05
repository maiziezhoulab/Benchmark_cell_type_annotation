# =======================
# SpaGCN runtime + memory profiling (FULL SCRIPT)
# =======================
# conda activate env_ST
import os
import csv
import re
import time
import psutil
import random
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import SpaGCN as spg

from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# -----------------------
# Helper: CPU memory (RSS)
# -----------------------
def get_cpu_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MiB


# -----------------------
# Load data
# -----------------------
rna = "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
adata = sc.read_h5ad(rna)

section_ids = [
    "0503_F5_T", "0503_F5_C", "0503_F5_L", "0503_F5_S", "0503_F4_C",
    "0503_F4_T", "0503_F3_C", "0503_F4_S", "0503_F3_L", "0503_F3_T",
    "0503_M5_C", "0503_F3_S", "0503_M5_T", "0503_M4_C", "0503_M5_S",
    "0503_M4_L", "0503_M4_T", "0503_M4_S"
]

# output directory
out_dir = "/maiziezhou_lab2/yuling/label_Transfer/SpaGCN/MouseSpinal"
os.makedirs(out_dir, exist_ok=True)

# record runtime & memory
records = []

# =======================
# Main loop
# =======================
for i, section in enumerate(section_ids):

    print(f"\n===== Processing section {section} ({i}) =====")

    ad = adata[adata.obs["Section ID"] == section, :].copy()

    # spatial coordinates
    x_pixel = ad.obsm["spatial"][:, 0].tolist()
    y_pixel = ad.obsm["spatial"][:, 1].tolist()

    adj = spg.calculate_adj_matrix(
        x=x_pixel,
        y=y_pixel,
        histology=False
    )

    # preprocessing
    ad.var_names_make_unique()
    spg.prefilter_genes(ad, min_cells=1)
    spg.prefilter_specialgenes(ad)

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    # SpaGCN parameters
    p = 0.5
    l = spg.search_l(
        p, adj,
        start=0.01,
        end=1000,
        tol=0.01,
        max_run=100
    )

    n_clusters = len(np.unique(ad.obs["MERFISH cell type annotation"]))

    r_seed = t_seed = n_seed = 2024
    res = spg.search_res(
        ad, adj, l, n_clusters,
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=r_seed,
        t_seed=t_seed,
        n_seed=n_seed
    )

    clf = spg.SpaGCN()
    clf.set_l(l)

    random.seed(r_seed)
    np.random.seed(n_seed)
    torch.manual_seed(t_seed)

    # =======================
    # START PROFILING
    # =======================
    start_time = time.time()
    cpu_mem_before = get_cpu_mem_mb()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- train ----
    clf.train(
        ad,
        adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200
    )

    # ---- predict ----
    y_pred, prob = clf.predict()

    # =======================
    # END PROFILING
    # =======================
    elapsed_time = time.time() - start_time
    cpu_mem_after = get_cpu_mem_mb()
    peak_cpu_mem = max(cpu_mem_before, cpu_mem_after)

    peak_gpu_mem = None
    if torch.cuda.is_available():
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2  # MiB

    # -----------------------
    # Save embeddings
    # -----------------------
    np.savetxt(
        f"{out_dir}/array{i}.csv",
        clf.embed,
        delimiter=",",
        fmt="%.6f"
    )

    # -----------------------
    # Save predictions
    # -----------------------
    ad.obs["pred"] = pd.Categorical(y_pred)

    adj_2d = spg.calculate_adj_matrix(
        x=x_pixel,
        y=y_pixel,
        histology=False
    )

    refined_pred = spg.refine(
        sample_id=ad.obs.index.tolist(),
        pred=ad.obs["pred"].tolist(),
        dis=adj_2d,
        shape="square"
    )

    ad.obs["refined_pred"] = pd.Categorical(refined_pred)

    ad.obs.to_csv(f"{out_dir}/ad_obs{i}.csv")

    # -----------------------
    # Log results
    # -----------------------
    records.append({
        "section": section,
        "runtime_sec": elapsed_time,
        "peak_cpu_mem_MiB": peak_cpu_mem,
        "peak_gpu_mem_MiB": peak_gpu_mem
    })

    print(f"Runtime (s): {elapsed_time:.2f}")
    print(f"Peak CPU memory (MiB): {peak_cpu_mem:.1f}")
    if peak_gpu_mem is not None:
        print(f"Peak GPU memory (MiB): {peak_gpu_mem:.1f}")

# =======================
# Save summary CSV
# =======================
runtime_df = pd.DataFrame(records)
runtime_df.to_csv(
    f"{out_dir}/runtimeSec_memoryMiB.csv",
    index=False
)

print("\n=== DONE ===")
print(runtime_df)

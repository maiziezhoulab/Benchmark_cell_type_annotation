# =========================================================
# Imports
# =========================================================
import os
import os.path as osp
import time
import gc
import psutil
import tracemalloc

from scipy import sparse
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scvi.model import CondSCVI, DestVI

import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# Global settings
# =========================================================
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")

process = psutil.Process(os.getpid())

# =========================================================
# Helper functions (profiling)
# =========================================================
def get_cpu_mem_mb():
    """Current process RSS memory in MB"""
    return process.memory_info().rss / 1024**2


def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    else:
        return np.nan


# =========================================================
# Paths & metadata
# =========================================================
save_dir = "/maiziezhou_lab2/yuling/label_Transfer/DestVI/Kidney_output"
os.makedirs(save_dir, exist_ok=True)

snRNA_time = ["Control", "4hours", "12hours", "2days", "14days", "6weeks"]
time_point = ["Sham", "Hour4", "Hour12", "Day2", "Day14", "Week6"]

# =========================================================
# Load data
# =========================================================
data = sc.read_h5ad(
    "/maiziezhou_lab2/yuling/Datasets/Kidney/snRNA_processed.h5ad"
)  # scRNA-seq

adata = sc.read_h5ad(
    "/maiziezhou_lab2/yuling/Datasets/Kidney/Xenium.h5ad"
)  # Spatial

# Ensure raw counts
adata.X = adata.raw.X.copy()
adata.layers["counts"] = adata.X.copy()

print(
    "SC counts stats:",
    "min =", data.X.min(),
    "max =", data.X.max(),
)

# =========================================================
# Profiling results container
# =========================================================
results = []

# =========================================================
# Main loop: time point × ident
# =========================================================
for k, tp in enumerate(time_point):

    print(f"\n================ {tp} ================\n")

    sn_data_full = data

    for ident in adata.obs["ident"].unique():

        tag = f"{tp}_{ident}"
        print(f"\n----- Running {tag} -----")

        # -------------------- profiling start --------------------
        gc.collect()
        tracemalloc.start()
        reset_gpu_peak()

        t_start = time.time()
        cpu_mem_start = get_cpu_mem_mb()
        # --------------------------------------------------------

        # -------------------- Prepare ST data -------------------
        st_data = adata[adata.obs["ident"] == ident].copy()
        st_data.layers["counts"] = st_data.layers.get(
            "counts", st_data.X.copy()
        )
        st_data.X = st_data.layers["counts"].astype("float32")

        # -------------------- Gene intersection -----------------
        common_genes = sn_data_full.var_names.intersection(
            st_data.var_names
        )
        print(f"Common genes: {len(common_genes)}")

        sn_data = sn_data_full[:, common_genes].copy()
        st_data = st_data[:, common_genes].copy()

        sn_data.layers["counts"] = sn_data.X.copy()
        st_data.layers["counts"] = st_data.X.copy()

        # -------------------- SC sanity checks ------------------
        X = sn_data.layers["counts"]
        X = X.A if sparse.issparse(X) else np.asarray(X)

        mask_finite = np.isfinite(X).all(axis=1)
        sn_data = sn_data[mask_finite].copy()

        lib_size = X.sum(axis=1)
        sn_data = sn_data[lib_size > 0].copy()

        print(
            f"Filtered SC: n_cells={sn_data.n_obs}, "
            f"n_genes={sn_data.n_vars}"
        )

        # -------------------- CondSCVI --------------------------
        scvi.settings.seed = 0
        CondSCVI.setup_anndata(
            sn_data,
            layer="counts",
            labels_key="name",  # SC cell type column
        )

        sc_model = CondSCVI(sn_data, weight_obs=False)
        sc_model.train(max_epochs=300, lr=1e-4)

        # -------------------- DestVI ----------------------------
        DestVI.setup_anndata(
            st_data,
            layer="counts",
        )

        st_model = DestVI.from_rna_model(st_data, sc_model)
        st_model.train(max_epochs=2500)

        # -------------------- Save proportions ------------------
        st_data.obsm["proportions"] = st_model.get_proportions()
        st_data.obsm["proportions"].to_csv(
            osp.join(save_dir, f"{tp}_{ident}_proportions.csv"),
            index=True,
        )

        # -------------------- profiling end ---------------------
        t_end = time.time()
        cpu_mem_end = get_cpu_mem_mb()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gpu_peak = get_gpu_peak_mb()

        results.append(
            {
                "time_point": tp,
                "ident": ident,
                "runtime_sec": t_end - t_start,
                "cpu_mem_peak_MB": peak / 1024**2,
                "cpu_mem_start_MB": cpu_mem_start,
                "cpu_mem_end_MB": cpu_mem_end,
                "gpu_mem_peak_MB": gpu_peak,
            }
        )
        df_profile = pd.DataFrame(results)
        df_profile.to_csv(
            osp.join(save_dir, "DestVI_time_memory_profile.csv"),
            index=False,
        )
        print(
            f"[{tag}] "
            f"time={t_end - t_start:.1f}s | "
            f"CPU peak={peak / 1024**2:.1f} MB | "
            f"GPU peak={gpu_peak:.1f} MB"
        )

        # -------------------- cleanup ---------------------------
        del sc_model, st_model, sn_data, st_data
        torch.cuda.empty_cache()
        gc.collect()

# =========================================================
# Save profiling summary
# =========================================================
df_profile = pd.DataFrame(results)
df_profile.to_csv(
    osp.join(save_dir, "DestVI_time_memory_profile.csv"),
    index=False,
)

print("\n=== PROFILING COMPLETE ===")
print(df_profile)
print("Saved to DestVI_time_memory_profile.csv")

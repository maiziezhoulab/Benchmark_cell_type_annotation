# ===================== Imports =====================
import os
import os.path as osp
import time
import tempfile

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse

import torch
import scvi
from scvi.model import SCVI, SCANVI

import seaborn as sns
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# ===================== Global settings =====================
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()

# ===================== Load data =====================
#snRNA_time = ['Control', '4hours', '12hours', '2days', '14days', '6weeks']
time_point = ['Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6']

data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Kidney/snRNA_processed.h5ad')
ad_sp = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Kidney/Xenium.h5ad')
ad_sp.obs['name'] = 'unknown'
# ensure counts
ad_sp.X = ad_sp.raw.X.todense()
ad_sp.layers["counts"] = ad_sp.X.copy()
data.layers["counts"] = data.X.copy()

# ===================== Sanity check (once, not benchmarked) =====================
X = data.layers["counts"]
X_dense = X.A if sparse.issparse(X) else np.asarray(X)

mask_finite = np.isfinite(X_dense).all(axis=1)
data = data[mask_finite].copy()

lib_size = X_dense.sum(axis=1)
mask_nonzero = lib_size > 0
data = data[mask_nonzero].copy()

data.X = data.layers["counts"].astype("float32")
data.layers["counts"] = data.X

# ===================== Model-only function (BENCHMARK CORE) =====================
def run_scvi_scanvi(adata):
    """
    Only SCVI + SCANVI setup / train / predict
    NO preprocessing inside this function
    """

    SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="tech"
    )

    scvi_model = SCVI(
        adata,
        n_layers=2,
        n_latent=30,
        gene_likelihood="nb",
        use_batch_norm="none",
        use_layer_norm="both"
    )
    scvi_model.train()

    scanvi_model = SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        labels_key="celltype_scanvi",
        unlabeled_category="Unknown"
    )
    scanvi_model.train(
        max_epochs=20,
        n_samples_per_label=100,
        batch_size=256
    )

    preds = scanvi_model.predict(adata)
    return preds


# ===================== Output dirs =====================
out_pred = '/maiziezhou_lab2/yuling/label_Transfer/scVI/Kidney_all'
os.makedirs(out_pred, exist_ok=True)
query = ["L", "R"]
# ===================== Benchmark loop =====================
records = []
sc_data = data
sc_data.obs['tech'] = 'sc'
for k, tp in enumerate(time_point):

    for ident in query:

        # ---------- PREPROCESSING (NOT BENCHMARKED) ----------
        st_data = ad_sp[ad_sp.obs['ident'] == tp + ident, :].copy()
        st_data.layers["counts"] = st_data.layers.get("counts", st_data.X.copy())
        st_data.X = st_data.layers["counts"].astype("float32")
        st_data.obs['tech'] = 'st'

        genes = st_data.var_names.intersection(sc_data.var_names)
        st_data = st_data[:, genes].copy()
        sc_sub = sc_data[:, genes].copy()

        adata = ad.concat([st_data, sc_sub])
        adata.layers["counts"] = adata.X.copy()

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=2000,
            layer="counts",
            batch_key="tech",
            subset=True
        )

        adata.layers["counts"] = adata.X.copy()

        adata.obs["celltype_scanvi"] = "Unknown"
        mask_sc = adata.obs["tech"] == "sc"
        adata.obs.loc[mask_sc, "celltype_scanvi"] = adata.obs.loc[mask_sc, "name"]

        # ---------- BENCHMARK START ----------
        torch.cuda.empty_cache()

        start = time.perf_counter()
        peak_mem, preds = memory_usage(
            (run_scvi_scanvi, (adata,)),
            retval=True,
            max_usage=True,
            interval=0.1
        )
        runtime = time.perf_counter() - start
        # ---------- BENCHMARK END ----------

        # save predictions
        pred = adata[adata.obs["celltype_scanvi"] == "Unknown"].copy()
        pred.obs["pred"] = preds[adata.obs["celltype_scanvi"] == "Unknown"]
        pred.obs.to_csv(
            osp.join(out_pred, f"{tp}_{ident}_label_transfer.csv")
        )

        # record benchmark
        records.append({
            "method": "scVI+scANVI",
            "time_point": tp,
            "ident": ident,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "runtime_sec": runtime,
            "peak_mem_mb": peak_mem
        })

        print(f"[DONE] {tp} | {ident} | time={runtime:.1f}s | mem={peak_mem:.1f}MB")

# ===================== Save benchmark table =====================
df_benchmark = pd.DataFrame(records)
df_benchmark.to_csv(
    osp.join(out_pred, "runtimeSec_memoryMiB.csv"),
    index=False
)

print("=== Benchmark finished ===")

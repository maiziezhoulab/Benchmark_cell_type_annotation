# ===============================
# GPU + TIME + MEMORY BENCHMARK
# SCVI + scANVI (combined)
# ===============================

import os
import time
import psutil
import torch
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvi

# -------------------------------
# Environment & reproducibility
# -------------------------------
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")

assert torch.cuda.is_available(), "CUDA is NOT available!"
print("Using GPU:", torch.cuda.get_device_name(0))

# -------------------------------
# Paths
# -------------------------------
outdir = "/maiziezhou_lab2/yuling/label_Transfer/scVI/HumanLymph"
os.makedirs(outdir, exist_ok=True)

# -------------------------------
# Load data function
# -------------------------------
def load_HMlymphNode(
    root_dir="/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad",
    section_id="1",
):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)
    adata = adataT[adataT.obs["n_section"] == section_id].copy()
    adata.obs["original_clusters"] = adata.obs["annotation"]
    adata.obs["batch"] = section_id
    if "gene_name" not in adata.var:
        adata.var["gene_name"] = adata.var_names
    return adata

# -------------------------------
# Load ST + scRNA-seq
# -------------------------------
slice_ids = ["2","3","4","5","6","7","9","11","17","18","19","23","24","25","26","28","33","34","36"]

st_data = load_HMlymphNode(section_id=slice_ids[10])
st_data.obs["tech"] = "st"

sc_data = load_HMlymphNode(section_id=slice_ids[4])
sc_data.obs["tech"] = "sc"
sc_data.layers["counts"] = sc_data.X.copy()

adata = anndata.concat([st_data, sc_data])
adata.layers["counts"] = adata.X.copy()

# -------------------------------
# Preprocessing
# -------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=2000,
    layer="counts",
    batch_key="tech",
    subset=True,
)

# -------------------------------
# Setup SCVI
# -------------------------------
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="tech",
)

# ===============================
# START BENCHMARK
# ===============================
torch.cuda.reset_peak_memory_stats()
start_time = time.perf_counter()

# -------- SCVI --------
scvi_model = scvi.model.SCVI(
    adata,
    n_layers=2,
    n_latent=30,
)

scvi_model.train(
    accelerator="gpu",
    devices=1,
)

# -------- scANVI --------
SCANVI_LABEL_KEY = "celltype_scanvi"
adata.obs[SCANVI_LABEL_KEY] = "Unknown"
mask_sc = adata.obs["tech"] == "sc"
adata.obs.loc[mask_sc, SCANVI_LABEL_KEY] = adata.obs.loc[mask_sc, "original_clusters"]

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    adata=adata,
    labels_key=SCANVI_LABEL_KEY,
    unlabeled_category="Unknown",
)

scanvi_model.train(
    max_epochs=20,
    n_samples_per_label=100,
    accelerator="gpu",
    devices=1,
)

# ===============================
# END BENCHMARK
# ===============================
total_time_sec = time.perf_counter() - start_time
peak_gpu_gb = torch.cuda.max_memory_allocated() / 1024**3

process = psutil.Process(os.getpid())
cpu_mem_gb = process.memory_info().rss / 1024**3

print(f"Total SCVI + scANVI time (sec): {total_time_sec:.2f}")
print(f"Peak GPU memory (GB): {peak_gpu_gb:.2f}")
print(f"CPU RAM usage (GB): {cpu_mem_gb:.2f}")

# -------------------------------
# Save results
# -------------------------------
bench_df = pd.DataFrame({
    "method": ["SCVI+scANVI"],
    "time_sec": [total_time_sec],
    "gpu_mem_gb": [peak_gpu_gb],
    "cpu_mem_gb": [cpu_mem_gb],
    "n_cells": [adata.n_obs],
    "n_genes": [adata.n_vars],
    "latent_dim": [30],
})

bench_df.to_csv(
    os.path.join(outdir, "runtime_memory_scvi_scanvi_total.csv"),
    index=False,
)

# -------------------------------
# Save AnnData
# -------------------------------
adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation()
adata.obs["C_scANVI"] = scanvi_model.predict(adata)

adata.write_h5ad(
    os.path.join(outdir, "adata_scANVI_embedding.h5ad"),
    compression="gzip"
)


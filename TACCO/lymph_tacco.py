import os
import sys
import time
import tracemalloc
import os.path as osp

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import tacco as tc
from sklearn.metrics import adjusted_rand_score

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
slice_ids = ["2", "3", "4", "5", "6", "7", "9", "11",
             "17", "18", "19", "23", "24", "25", "26",
             "28", "33", "34", "36"]

def load_HMlymphNode(
    root_dir='/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad',
    section_id="1"
):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)
    slice1 = adataT[adataT.obs['n_section'] == section_id].copy()
    if 'gene_name' not in slice1.var.columns:
        slice1.var['gene_name'] = slice1.var_names
    slice1.obs['original_clusters'] = slice1.obs['annotation']
    slice1.obs['batch'] = section_id
    return slice1


def make_float32(adata):
    if adata.is_view:
        adata = adata.copy()

    if getattr(adata, "isbacked", False):
        adata = ad.read_h5ad(adata.filename, backed=None)

    if sp.issparse(adata.X):
        adata.X = adata.X.asfptype().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata


# -------------------------------------------------
# Load data
# -------------------------------------------------
reference_data = load_HMlymphNode(section_id=slice_ids[4])
puck = load_HMlymphNode(section_id=slice_ids[10])

reference_data = make_float32(reference_data)
puck = make_float32(puck)

# If counts stored in layers, move to X
if 'counts' in puck.layers:
    L = puck.layers['counts']
    puck.X = L.asfptype().astype(np.float32) if sp.issparse(L) else np.asarray(L, dtype=np.float32)

if 'counts' in reference_data.layers:
    L = reference_data.layers['counts']
    reference_data.X = L.asfptype().astype(np.float32) if sp.issparse(L) else np.asarray(L, dtype=np.float32)

print("puck.X dtype:", puck.X.dtype)
print("reference.X dtype:", reference_data.X.dtype)

# -------------------------------------------------
# Benchmark: runtime + peak memory
# -------------------------------------------------
tracemalloc.start()
start_time = time.perf_counter()

# ------------------
# TACCO annotation
# ------------------
tc.tl.annotate(
    puck,
    reference_data,
    annotation_key='annotation',
    result_key='ClusterName',
    counts_location='X',
    assume_valid_counts=True
)

end_time = time.perf_counter()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime_sec = end_time - start_time
peak_mem_mib = peak_mem / 1024**2

print(f"Runtime (sec): {runtime_sec:.2f}")
print(f"Peak CPU memory (MiB): {peak_mem_mib:.2f}")

# -------------------------------------------------
# Post-processing: ACC / ARI
# -------------------------------------------------
tmp = puck.obsm['ClusterName']

max_col = tmp.idxmax(axis=1).rename('max_col')
max_val = tmp.max(axis=1).rename('max_val')

out = pd.concat([max_col, max_val], axis=1)
out.index = out.index.astype(puck.obs.index.dtype)

merged = puck.obs.join(out, how='left')

a = merged['original_clusters'].astype('category')
b = merged['max_col'].astype('category')

cats = a.cat.categories.union(b.cat.categories)
a = a.cat.set_categories(cats)
b = b.cat.set_categories(cats)

ari = adjusted_rand_score(merged['original_clusters'], merged['max_col'])
acc = (a == b).sum() / len(merged)

# -------------------------------------------------
# Save results
# -------------------------------------------------
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/tacco/HumanLymph_output"
os.makedirs(outdir, exist_ok=True)

merged.to_csv(osp.join(outdir, "Results.csv"))

summary = pd.DataFrame([{
    "Elapsed_Time_sec": runtime_sec,
    "Peak_RAM_Used_MiB": peak_mem_mib,
}])

summary.to_csv(
    osp.join(outdir, "runtimeSec_memoryMiB.csv"),
    index=False
)

print(summary)

import os
import sys
import tracemalloc
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc 
import tacco as tc
import time 
# The notebook expects to be executed either in the workflow directory or in the repository root folder...
sys.path.insert(1, os.path.abspath('workflow' if os.path.exists('workflow/common_code.py') else '..')) 
import common_code
import numpy as np
import scipy.sparse as sp

def make_float32(adata):
    # If it's a view, materialize it
    if adata.is_view:
        adata = adata.copy()

    # If the AnnData was opened in backed mode, reload into memory
    if getattr(adata, "isbacked", False):
        import anndata as ad
        adata = ad.read_h5ad(adata.filename, backed=None)

    # Convert X
    if sp.issparse(adata.X):
        adata.X = adata.X.asfptype()           # to float
        adata.X = adata.X.astype(np.float32)   # to float32
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    return adata
import os.path as osp 
import scanpy as sc
 
rows = []
 
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/tacco/tacco_examples/Kidney_output"
os.makedirs(outdir, exist_ok=True)
query = ['L', 'R']
time_point = ['Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6']

# =========================
# Main loop
# =========================
for i in time_point:
    for j in query:

        print(f"\n=== TACCO | {i}{j} ===")

        # -------------------------
        # Load data (NOT timed)
        # -------------------------
        puck = sc.read_h5ad(
            f"/maiziezhou_lab2/yuling/datasets/Kidney/Xenium/time_{i}{j}.h5ad"
        )
        puck.X = puck.raw.X

        adata_subset = sc.read_h5ad(
            f"/maiziezhou_lab2/yuling/datasets/Kidney/snRNA-seq/time_{i}.h5ad"
        )

        puck = make_float32(puck)
        adata_subset = make_float32(adata_subset)

        adata_subset.obs['celltype'] = adata_subset.obs['name']

        mapping = {
            # TAL
            "MTAL": "TAL",
            "CTAL1": "TAL",
            "CTAL2": "TAL",

            # CNT
            "CNT": "CNT",
            "DCT-CNT": "CNT",

            # EC
            "EC1": "EC",
            "EC2": "EC",

            # PC
            "PC1": "PC",
            "PC2": "PC",

            # PT
            "NewPT1": "Inj_PT",
            "NewPT2": "FR_PT",

            # Immune
            "Mø": "Immune",
            "Tcell": "Immune",
        }
        adata_subset.obs["celltype"] = adata_subset.obs["celltype"].replace(mapping)

        # If counts exist in layers, move to X
        if 'counts' in puck.layers:
            L = puck.layers['counts']
            puck.X = L.asfptype().astype(np.float32) if sp.issparse(L) else np.asarray(L, dtype=np.float32)

        if 'counts' in adata_subset.layers:
            L = adata_subset.layers['counts']
            adata_subset.X = L.asfptype().astype(np.float32) if sp.issparse(L) else np.asarray(L, dtype=np.float32)

        # -------------------------
        # Model profiling starts
        # -------------------------
        tracemalloc.start()
        t_start = time.perf_counter()

        tc.tl.annotate(
            puck,
            adata_subset,
            'celltype',
            result_key='ClusterName',
            counts_location='X',
            assume_valid_counts=True
        )

        t_end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime_sec = t_end - t_start
        peak_mem_mb = peak / 1024 / 1024

        # -------------------------
        # Evaluation (NOT timed)
        # -------------------------
        tmp = puck.obsm['ClusterName']

        max_col = tmp.idxmax(axis=1).rename('max_col')
        max_val = tmp.max(axis=1).rename('max_val')

        out = pd.concat([max_col, max_val], axis=1)
        out.index = out.index.astype(puck.obs.index.dtype)

        merged = puck.obs.join(out, how='left')

        a = merged["celltype_plot"].astype('category')
        b = merged["max_col"].astype('category')

        cats = a.cat.categories.union(b.cat.categories)
        a = a.cat.set_categories(cats)
        b = b.cat.set_categories(cats)

        # -------------------------
        # Save results
        # -------------------------
        rows.append({
            "method": "TACCO",
            "dataset": "Xenium",
            "time_point": i,
            "side": j,
            "runtime_sec": runtime_sec,
            "peak_mem_mb": peak_mem_mb
        })

        outfile = osp.join(outdir, f"{i}{j}_pred.csv")
        merged.to_csv(outfile)

        print(f"Runtime: {runtime_sec:.2f}s | Peak memory: {peak_mem_mb:.1f} MB")


# =========================
# Save benchmark table
# =========================
df_bench = pd.DataFrame(rows)
df_bench.to_csv(osp.join(outdir, "tacco_benchmark_summary.csv"), index=False)
print("\nSaved benchmark summary.")
import os
import os.path as osp
import time
import tracemalloc

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import tangram as tg


# =========================
# Utils
# =========================
def make_float32(adata):
    if sp.issparse(adata.X):
        adata.X = adata.X.asfptype().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


# =========================
# Config
# =========================
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/Kidney_all_output"
os.makedirs(outdir, exist_ok=True)

query = ['L', 'R']
time_point = ['Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6']

rows = []


# =========================
# Main loop
# =========================
for i in time_point:
    for j in query:

        print(f"\n=== Tangram | {i}{j} ===")

        # -------------------------
        # Load & preprocessing (NOT timed)
        # -------------------------
        reference_data = sc.read_h5ad(
            "/maiziezhou_lab2/yuling/datasets/Kidney/snRNA_cleaned.h5ad"
        )
        reference_data.obs['celltype'] = reference_data.obs['name']

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
        reference_data.obs["celltype"] = reference_data.obs["celltype"].replace(mapping)

        query_data = sc.read_h5ad(
            f"/maiziezhou_lab2/yuling/datasets/Kidney/Xenium/time_{i}{j}.h5ad"
        )
        query_data.X = query_data.raw.X

        # Normalization / HVG
        sc.pp.normalize_total(query_data)
        sc.pp.log1p(query_data)

        sc.pp.normalize_total(reference_data)
        sc.pp.log1p(reference_data)
        sc.pp.highly_variable_genes(
            reference_data,
            n_top_genes=3000,
            flavor="seurat_v3"
        )

        hvg = reference_data.var_names[
            reference_data.var['highly_variable']
        ].to_list()

        genes = list(set(hvg).intersection(set(query_data.var_names)))

        reference_data = reference_data[:, genes].copy()
        query_data = query_data[:, genes].copy()

        # Harmonize genes (still preprocessing)
        tg.pp_adatas(reference_data, query_data, genes=None)

        reference_data = make_float32(reference_data)
        query_data = make_float32(query_data)

        # -------------------------
        # Model profiling starts
        # -------------------------
        tracemalloc.start()
        t_start = time.perf_counter()

        tg_map = tg.map_cells_to_space(
            reference_data,
            query_data,
            density_prior='uniform',
            device='cpu'   # GPU if needed
        )

        tg.project_cell_annotations(
            adata_sp=query_data,
            adata_map=tg_map,
            annotation='celltype'
        )

        t_end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime_sec = t_end - t_start
        peak_mem_mb = peak / 1024 / 1024

        # -------------------------
        # Save results (NOT timed)
        # -------------------------
        df = pd.DataFrame(
            query_data.obsm['tangram_ct_pred'],
            index=query_data.obs_names
        )

        outfile = osp.join(outdir, f"{i}{j}_pred.csv")
        df.to_csv(outfile)

        rows.append({
            "method": "Tangram",
            "dataset": "Xenium",
            "time_point": i,
            "side": j,
            "n_spots": query_data.n_obs,
            "n_genes": query_data.n_vars,
            "runtime_sec": runtime_sec,
            "peak_mem_mb": peak_mem_mb
        })

        print(f"Runtime: {runtime_sec:.2f}s | Peak memory: {peak_mem_mb:.1f} MB")
        print(f"Saved result: {outfile}")


# =========================
# Save benchmark table
# =========================
df_bench = pd.DataFrame(rows)
df_bench.to_csv(
    osp.join(outdir, "runtimeSec_memoryMiB.csv"),
    index=False
)
print("\nSaved Tangram benchmark summary.")

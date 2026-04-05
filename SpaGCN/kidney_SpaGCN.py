import os, time, tracemalloc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
import random, torch

warnings.filterwarnings("ignore")

############################################
# Global settings
############################################
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

sc.set_figure_params(figsize=(6, 6), frameon=False)
torch.set_float32_matmul_precision("high")

############################################
# Paths
############################################
save_dir = "/maiziezhou_lab2/yuling/label_Transfer/SpaGCN/kidney"
os.makedirs(save_dir, exist_ok=True)

############################################
# Load data (NOT benchmarked)
############################################
adata = sc.read_h5ad("/maiziezhou_lab2/yuling/Datasets/Kidney/Xenium.h5ad")

adata.X = adata.raw.X.copy()
adata.layers["counts"] = adata.X.copy()

############################################
# Benchmark results container
############################################
benchmark_records = []

############################################
# Loop over sections
############################################
for m in adata.obs["ident"].unique():

    print(f"\n=== Running SpaGCN for kidney section: {m} ===")

    ad = adata[adata.obs["ident"] == m, :].copy()

    # ---------- preprocessing (NOT benchmarked) ----------
    x_pixel = ad.obsm["spatial"][:, 0].tolist()
    y_pixel = ad.obsm["spatial"][:, 1].tolist()

    adj = spg.calculate_adj_matrix(
        x=x_pixel, y=y_pixel, histology=False
    )

    ad.var_names_make_unique()
    spg.prefilter_genes(ad, min_cells=1)
    spg.prefilter_specialgenes(ad)

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    p = 0.5
    l = spg.search_l(
        p, adj, start=0.01, end=1000, tol=0.01, max_run=100
    )

    n_clusters = len(np.unique(ad.obs["celltype_plot"]))

    ############################################
    # Model stage (BENCHMARKED)
    ############################################
    tracemalloc.start()
    t_start = time.time()

    # search resolution
    res = spg.search_res(
        ad, adj, l, n_clusters,
        start=0.7, step=0.1, tol=5e-3,
        lr=0.05, max_epochs=20,
        r_seed=SEED, t_seed=SEED, n_seed=SEED
    )

    clf = spg.SpaGCN()
    clf.set_l(l)

    clf.train(
        ad, adj,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200
    )

    y_pred, prob = clf.predict()
    ad.obs["pred"] = y_pred.astype(str)

    adj_2d = spg.calculate_adj_matrix(
        x=x_pixel, y=y_pixel, histology=False
    )

    refined_pred = spg.refine(
        sample_id=ad.obs.index.tolist(),
        pred=ad.obs["pred"].tolist(),
        dis=adj_2d,
        shape="square"
    )

    ad.obs["refined_pred"] = refined_pred

    elapsed_sec = time.time() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_MiB = peak_bytes / (1024 ** 2)

    ############################################
    # Save outputs
    ############################################
    ad.obs.to_csv(
        os.path.join(save_dir, f"kidney_{m}_obs.csv")
    )

    benchmark_records.append({
        "method": "SpaGCN",
        "section": m,
        "seed": SEED,
        "Elapsed_Time_sec": elapsed_sec,
        "Peak_RAM_Used_MiB": peak_mem_MiB
    })

    print(
        f"Finished {m}: "
        f"time={elapsed_sec:.2f}s, "
        f"peak_mem={peak_mem_MiB:.1f} MiB"
    )

############################################
# Save benchmark summary
############################################
benchmark_df = pd.DataFrame(benchmark_records)

benchmark_df.to_csv(
    os.path.join(save_dir, "runtimeSec_memoryMiB.csv"),
    index=False
)

print("\n=== SpaGCN benchmarking finished ===")
print(benchmark_df)


import os, csv, re, time, random, warnings, resource
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import torch
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt

data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development/5DPIs.h5ad')
base_dir = '/maiziezhou_lab2/yuling/label_Transfer/SpaGCN/regeneration'
ad = data[data.obs['Batch'] == 'Injury_5DPI_rep2_SS200000147BL_D2',].copy()

os.makedirs(base_dir, exist_ok=True)
x_pixel = ad.obsm['spatial'][:, 0].tolist()
y_pixel = ad.obsm['spatial'][:, 1].tolist()
adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)

ad.var_names_make_unique()
spg.prefilter_genes(ad, min_cells=1)
spg.prefilter_specialgenes(ad)

sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)

p = 0.5
l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

ad.obs['Annotation'] = ad.obs['Annotation'].astype('category')
n_clusters = len(np.unique(ad.obs["Annotation"]))

seed = 2026
series_dir = '/maiziezhou_lab2/yuling/label_Transfer/SpaGCN/regeneration'
os.makedirs(series_dir, exist_ok=True)

time_rows = []

# -----------------------------
# set seeds
# -----------------------------
if seed is not None:
    r_seed = t_seed = n_seed = seed
    random.seed(r_seed)
    np.random.seed(n_seed)
    torch.manual_seed(t_seed)

# -----------------------------
# timing start
# -----------------------------
t0 = time.perf_counter()

# -----------------------------
# search resolution
# -----------------------------
res = spg.search_res(
    ad, adj, l, n_clusters,
    start=0.7, step=0.1,
    tol=5e-3, lr=0.05, max_epochs=20,
    r_seed=(seed if seed is not None else None),
    t_seed=(seed if seed is not None else None),
    n_seed=(seed if seed is not None else None),
)

# -----------------------------
# train SpaGCN
# -----------------------------
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

# predict
y_pred, prob = clf.predict()
ad.obs["pred"] = y_pred
ad.obs["pred"] = ad.obs["pred"].astype("category")

# -----------------------------
# refinement
# -----------------------------
adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
refined_pred = spg.refine(
    sample_id=ad.obs.index.tolist(),
    pred=ad.obs["pred"].tolist(),
    dis=adj_2d,
    shape="square"
)
ad.obs["refined_pred"] = refined_pred
ad.obs["refined_pred"] = ad.obs["refined_pred"].astype("category")

# -----------------------------
# timing / peak memory end
# -----------------------------
t1 = time.perf_counter()
elapsed = t1 - t0

# Linux: ru_maxrss is in KB
peak_mem_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# -----------------------------
# save obs
# -----------------------------
ad.obs.to_csv(os.path.join(series_dir, "Results.csv"))

# -----------------------------
# save runtime / peak memory
# -----------------------------
time_rows.append({
    "runtime_sec": elapsed,
    "peak_mem_MiB": peak_mem_mib
})

pd.DataFrame(time_rows).to_csv(
    os.path.join(series_dir, "runtimeSec_memoryMiB.csv"),
    index=False
)
import os, time, tracemalloc, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
import random, torch

warnings.filterwarnings("ignore")

############################################
# Fixed seed
############################################
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############################################
# Load data
############################################
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development.h5ad')

stage_44 = data[
    data.obs['Batch'] == 'Stage44_telencephalon_rep2_FP200000239BL_E4',
].copy()
stage_44.obs['Annotation'] = stage_44.obs['Annotation'].astype('category')
ad = stage_44

############################################
# Output directory
############################################
base_dir = '/maiziezhou_lab2/yuling/label_Transfer/SpaGCN/Development'
os.makedirs(base_dir, exist_ok=True)

############################################
# Preprocessing (NOT benchmarked)
############################################
x_pixel = ad.obsm['spatial'][:, 0].tolist()
y_pixel = ad.obsm['spatial'][:, 1].tolist()

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
    p, adj,
    start=0.01, end=1000,
    tol=0.01, max_run=100
)

n_clusters = len(np.unique(ad.obs['Annotation']))

############################################
# SpaGCN model stage (BENCHMARKED)
############################################
tracemalloc.start()
t_start = time.time()

# search resolution
res = spg.search_res(
    ad, adj, l, n_clusters,
    start=0.7, step=0.1,
    tol=5e-3, lr=0.05,
    max_epochs=20,
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
ad.obs["pred"] = pd.Categorical(y_pred)

adj_2d = spg.calculate_adj_matrix(
    x=x_pixel, y=y_pixel, histology=False
)

refined_pred = spg.refine(
    sample_id=ad.obs.index.tolist(),
    pred=ad.obs["pred"].tolist(),
    dis=adj_2d,
    shape="square"
)

ad.obs["refined_pred"] = pd.Categorical(refined_pred)

############################################
# End benchmark
############################################
elapsed_sec = time.time() - t_start
_, peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()

peak_mem_MiB = peak_bytes / (1024 ** 2)

############################################
# Save outputs
############################################
np.savetxt(
    os.path.join(base_dir, "slice_id_44_array.csv"),
    clf.embed,
    delimiter=",",
    fmt="%.6f"
)

ad.obs.to_csv(
    os.path.join(base_dir, "slice_id_44_ad_obs.csv")
)

summary_df = pd.DataFrame([{
    "runtime_sec": elapsed_sec,
    "peak_mem_MiB": peak_mem_MiB
}])

summary_df.to_csv(
    os.path.join(base_dir, "runtimeSec_memoryMiB.csv"),
    index=False
)

############################################
# Print summary
############################################
print("=== SpaGCN Development (Stage44) finished ===")
print(f"Runtime (sec): {elapsed_sec:.2f}")
print(f"Peak CPU memory (MiB): {peak_mem_MiB:.1f}")

import scanpy as sc
import os
import torch
import pandas as pd
from sklearn import metrics
from GraphST import GraphST
from GraphST.preprocess import filter_with_overlap_gene
import time
import anndata as ad
import random
import numpy as np
import sys, pathlib

# ---------------- R env ----------------
def set_r_env(prefix: str = None):
    if prefix is None:
        prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    prefix = str(pathlib.Path(prefix).resolve())
    os.environ["R_HOME"] = f"{prefix}/lib/R"
    os.environ["R_LIBS_USER"] = f"{prefix}/lib/R/library"
    os.environ["PATH"] = f"{prefix}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

set_r_env()

# ---------------- fixed seed (single run) ----------------
SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- load data ----------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development.h5ad')

stage_54 = data[data.obs['Batch'] == 'Stage54_telencephalon_rep2_DP8400015649BRD6_2'].copy()
stage_44 = data[data.obs['Batch'] == 'Stage44_telencephalon_rep2_FP200000239BL_E4'].copy()

# =========================================================
# 🔹 Preprocessing (NOT timed)
# =========================================================
adata_st = stage_44.copy()
GraphST.preprocess(adata_st)
GraphST.construct_interaction(adata_st)
GraphST.add_contrastive_label(adata_st)

adata_sc = stage_54.copy()
adata_sc.var_names_make_unique()
GraphST.preprocess(adata_sc)

adata_base, adata_sc_base = filter_with_overlap_gene(adata_st, adata_sc)
GraphST.get_feature(adata_base)

# 必须 copy
adata = adata_base.copy()
adata_sc = adata_sc_base.copy()

# =========================================================
# ⏱ Runtime + Memory (ONLY model pipeline)
# =========================================================
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

t_start = time.perf_counter()

# ---------------- Train + Inference ----------------
model = GraphST.GraphST(
    adata,
    adata_sc,
    epochs=1200,
    random_seed=SEED,
    device='cuda',
    deconvolution=True
)

adata, adata_sc = model.train_map()

from GraphST.utils import project_cell_to_spot
adata_sc.obs['cell_type'] = adata_sc.obs['Annotation']
project_cell_to_spot(
    adata,
    adata_sc,
    retain_percent=0.15
)

torch.cuda.synchronize()
runtime_sec = time.perf_counter() - t_start
peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

# =========================================================
# 🔹 Evaluation (NOT timed)
# =========================================================
last60 = adata.obs.iloc[:, 4:].apply(pd.to_numeric, errors="coerce")
adata.obs["top_col"] = last60.idxmax(axis=1)
adata.obs["top_value"] = last60.max(axis=1, skipna=True)

ari = metrics.adjusted_rand_score(
    adata.obs["top_col"],
    adata.obs["Annotation"]
)
acc = (adata.obs["top_col"] == adata.obs["Annotation"]).mean()

# ---------------- save outputs ----------------
out_dir = "/maiziezhou_lab2/yuling/label_Transfer/GraphST/Development"
os.makedirs(out_dir, exist_ok=True)

adata.obs.to_csv(
    f"{out_dir}/GraphST_output_1.csv",
    index=True,
    index_label="cell_id"
)

runtime_df = pd.DataFrame([{
   
    "Elapsed_Time_sec": runtime_sec,
    "Peak_RAM_Used_MiB": peak_mem_mib,
}])

runtime_df.to_csv(
    f"{out_dir}/runtimeSec_memoryMiB.csv",
    index=False
)

print(runtime_df)

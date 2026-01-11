import torch
import torch.nn as nn
import time
import sys
import os
import os.path as osp
import pandas as pd
import scanpy as sc
import anndata as ad

# ------------------------------------------------------------------
# path 
# ------------------------------------------------------------------
sys.path.append('/maiziezhou_lab2/yuling/label_Transfer/spatialID/SpatialID/spatialid')
from spatialid.transfer import Transfer

# ------------------------------------------------------------------
# set seed 
# ------------------------------------------------------------------
SEED = 2024
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------
base_dir = "/maiziezhou_lab2/yuling/label_Transfer/spatialID/dataset/Development"
os.makedirs(base_dir, exist_ok=True)

spatial_path = osp.join(base_dir, "spatial_data.h5ad")
single_cell_dir = osp.join(base_dir, "single_run")
os.makedirs(single_cell_dir, exist_ok=True)
single_cell_path = osp.join(single_cell_dir, "single_cell_data.h5ad")

# ------------------------------------------------------------------
# load data 
# ------------------------------------------------------------------
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/Development.h5ad')

# reference (scRNA-seq)
sc_adata = data[data.obs['Batch'] == 'Stage54_telencephalon_rep2_DP8400015649BRD6_2'].copy()

# query (spatial)
query_data = data[data.obs['Batch'] == 'Stage44_telencephalon_rep2_FP200000239BL_E4'].copy()

# 
query_data.write(spatial_path)
sc_adata.write_h5ad(single_cell_path)

# ------------------------------------------------------------------
# ⏱ Runtime + 🧠 Peak GPU memory（ONLY model pipeline）
# ------------------------------------------------------------------
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

t_start = time.perf_counter()

# ------------------------------------------------------------------
# SpatialID pipeline
# ------------------------------------------------------------------
transfer_tool = Transfer(
    spatial_data=spatial_path,
    single_data=single_cell_path,
    output_path=single_cell_dir,
    device=0          # GPU id, -1 for CPU
)

# Step 1: train sc model
transfer_tool.learn_sc(
    filter_mt=True,
    min_cell=0,
    min_gene=0,
    max_cell=98.0,
    ann_key="Annotation",
    batch_size=409,
    epoch=200,
    lr=3e-4
)

# Step 2: sc → st
transfer_tool.sc2st()

# Step 3: GNN-based annotation
transfer_tool.annotation(
    pca_dim=200,
    n_neigh=30,
    epochs=200,
    lr=0.01,
    show_results=True
)

torch.cuda.synchronize()
runtime_sec = time.perf_counter() - t_start
peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

print("SpatialID completed.")

# ------------------------------------------------------------------
# save runtime + memory
# ------------------------------------------------------------------
runtime_df = pd.DataFrame([{
   
    "Elapsed_Time_sec": runtime_sec,
    "Peak_RAM_Used_MiB": peak_mem_mib
}])

runtime_df.to_csv(
    osp.join(base_dir, "runtimeSec_memoryMiB_dnn.csv"),
    index=False
)

print(runtime_df)

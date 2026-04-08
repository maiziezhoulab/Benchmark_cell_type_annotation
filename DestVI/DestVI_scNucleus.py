import os
import tempfile
# conda activate DestVI
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import os.path as osp 
# import destvi_utils
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvi
import seaborn as sns
import torch
from scvi.model import CondSCVI, DestVI
import seaborn as sns
import torch
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = '/maiziezhou_lab2/yuling/label_Transfer/DestVI/scNucleus'
data = sc.read_h5ad('/maiziezhou_lab2/yuling/Datasets/obj_integrated_sc_nucleus.h5ad')
adata = sc.read_h5ad('/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad')
st_data = adata[adata.obs['Section ID'] == '0503_F4_C',] 
st_data.layers['counts'] = st_data.X
sc_adata = data 
sc_adata.layers['counts'] = sc_adata.X
scvi.settings.seed = 0
 # 2. find common genes 
common_genes = sc_adata.var_names.intersection(st_data.var_names)
print(f"Common genes: {len(common_genes)}")

sc_data = sc_adata[:, common_genes].copy()
st_data = st_data[:, common_genes].copy()

print(f"\nAfter filtering:")
print(f"scRNA-seq genes: {sc_data.n_vars}")
print(f"Spatial genes: {st_data.n_vars}")

CondSCVI.setup_anndata(sc_data, layer = "counts", labels_key = "final_cluster_assignment")
sc_model = CondSCVI(sc_data, weight_obs=False)
sc_model.view_anndata_setup()
sc_model.train()
DestVI.setup_anndata(st_data, layer = "counts")
st_model = DestVI.from_rna_model(st_data, sc_model)
st_model.view_anndata_setup()
st_model.train(max_epochs=2500)
st_data.obsm["proportions"] = st_model.get_proportions()
df = st_data.obsm["proportions"].copy()
df.to_csv(osp.join(save_dir,"proportions.csv"), index=True) 
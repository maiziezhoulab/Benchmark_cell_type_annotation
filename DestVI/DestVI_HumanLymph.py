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
import time
scvi.settings.seed = 0
sc.set_figure_params(figsize=(6, 6), frameon=False)
sns.set_theme()
torch.set_float32_matmul_precision("high")
save_dir = '/maiziezhou_lab2/yuling/label_Transfer/DestVI/HumanLymph'
slice_ids = ["2", "3", "4", "5", "6", "7", "9", "11", "17", "18", "19", "23", "24", "25", "26", "28", "33", "34", "36"]
def load_HMlymphNode(root_dir = '/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad', section_id =  "1"):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)  # Convert section_id to integer
    slice1 = adataT[adataT.obs['n_section'] == section_id]
    if 'gene_name' not in slice1.var.columns:
        slice1.var['gene_name'] = slice1.var_names
    slice1.obs['original_clusters'] = slice1.obs['annotation']
    slice1.obs['batch'] = section_id
    return slice1
section_ids = [4, 10]
 #----------------
sc_adata =load_HMlymphNode(section_id= slice_ids[4])
# 创建目录 series_k
scvi.settings.seed = 0
series_dir = '/maiziezhou_lab2/yuling/label_Transfer/DestVI/HumanLymph'
#adata = sc.read_h5ad('/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad')
st_data = load_HMlymphNode(section_id = slice_ids[10]) 
st_data.layers['counts'] = st_data.X
sc_adata.layers['counts'] = sc_adata.X
CondSCVI.setup_anndata(sc_adata, layer = "counts", labels_key = "original_clusters")
sc_model = CondSCVI(sc_adata, weight_obs=False)
sc_model.view_anndata_setup()
DestVI.setup_anndata(st_data, layer="counts")
st_model = DestVI.from_rna_model(st_data, sc_model)
st_model.view_anndata_setup()
#---------------------------------------------
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.perf_counter()
sc_model.train()
st_model.train(max_epochs=2500)
torch.cuda.synchronize()
end_time = time.perf_counter()

runtime_sec = end_time - start_time
peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

print(f"DestVI running time (GPU, sec): {runtime_sec:.2f}")
print(f"DestVI peak GPU memory (MB): {peak_mem_mb:.2f}")
st_data.obsm["proportions"] = st_model.get_proportions()
df = st_data.obsm["proportions"].copy()
df.to_csv(osp.join(series_dir,"proportions.csv"), index=True)  # keeps spot barcodes as the first column
pd.DataFrame([{
    "method": "DestVI",
    "runtime_sec": runtime_sec,
    "peak_mem_mb": peak_mem_mb
}]).to_csv(
    osp.join(series_dir, "destvi_runtime_memory.csv"),
    index=False
)

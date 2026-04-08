import tangram as tg
import scanpy as sc
import numpy as np
import pandas as pd
import os
import time
import torch
outdir = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/HumanLymph_output"
# conda activate loki_env
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
reference_data = load_HMlymphNode(section_id= slice_ids[4])
query_data = load_HMlymphNode(section_id= slice_ids[10])
# Preprocessing
sc.pp.normalize_total(query_data)
sc.pp.log1p(query_data)

sc.pp.normalize_total(reference_data)
sc.pp.log1p(reference_data)
sc.pp.highly_variable_genes(reference_data, n_top_genes=3000, flavor="seurat_v3")
hvg = reference_data.var_names[reference_data.var['highly_variable']].to_list()
genes = list(set(hvg).intersection(set(query_data.var_names)))
reference_data = reference_data[:, genes].copy()
query_data = query_data[:, genes].copy()
# Trim reference (Tangram is heavy on GPU/CPU memory)
#reference_data = reference_data[:25000].copy()

# Harmonize genes
tg.pp_adatas(reference_data, query_data, genes=None)

# ----------------------------
# Measure GPU time + memory
# ----------------------------
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.perf_counter()

tg_map = tg.map_cells_to_space(
    reference_data,
    query_data,
    density_prior='uniform',
    device='cuda'
)

torch.cuda.synchronize()
end_time = time.perf_counter()

runtime_sec = end_time - start_time
peak_mem_mib = torch.cuda.max_memory_allocated() / 1024**2

print(f"Tangram runtime (sec): {runtime_sec:.2f}")
print(f"Tangram GPU peak memory (MiB): {peak_mem_mib:.2f}")


'''
tg_map = tg.map_cells_to_space(
    reference_data,
    query_data,
    density_prior='uniform',
    device='cuda'   # GPU if enough VRAM
)
'''
# Project cell type annotation (make sure this obs column exists!)
tg.project_cell_annotations(
    adata_sp=query_data,
    adata_map=tg_map,
    annotation='original_clusters'
)

# Save results
df = pd.DataFrame(query_data.obsm['tangram_ct_pred'], index=query_data.obs_names)
outfile = os.path.join(outdir, "tangram_pred.csv")
df.to_csv(outfile)
print(f"Saved result: {outfile}")

benchmark_df = pd.DataFrame({
    "Elapsed_Time_sec": [runtime_sec],
    "Peak_RAM_Used_MiB": [peak_mem_mib]
})

benchmark_df.to_csv(
    "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Tangram/HumanLymph_output/runtimeSec_memoryMiB.csv",
    index=False
)
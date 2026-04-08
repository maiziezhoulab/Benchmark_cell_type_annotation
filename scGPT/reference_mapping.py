from pathlib import Path
import numpy as np
from scipy.stats import mode
import scanpy as sc
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import sys

sys.path.insert(0, "../")

import scgpt as scg

try:
    import faiss

    faiss_imported = True
except ImportError:
    faiss_imported = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")

warnings.filterwarnings("ignore", category=ResourceWarning) 
model_dir = Path("/maiziezhou_lab2/yuling/scGPT_model/scGPT_human")

cell_type_key = "original_clusters"
gene_col = "gene_name"

# load and pre-process data 
slice_ids = ["2", "3", "4", "5", "6", "7", "9", "11", "17", "18", "19", "23", "24", "25", "26", "28", "33", "34", "36"]
def load_HMlymphNode(root_dir = '/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad', section_id =  "1"):
    adataT = sc.read_h5ad(root_dir)
    section_id = int(section_id)  # Convert section_id to integer
    slice1 = adataT[adataT.obs['n_section'] == section_id]
    if 'gene_name' not in slice1.var.columns:
        slice1.var['gene_name'] = slice1.var_names
    slice1.obs['original_clusters'] = slice1.obs['annotation']
    return slice1
adata =load_HMlymphNode(section_id=slice_ids[4])
test_adata = load_HMlymphNode(section_id=slice_ids[10])
ref_embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)
test_embed_adata = scg.tasks.embed_data(
    test_adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)
# concatenate the two datasets
adata_concat = test_embed_adata.concatenate(ref_embed_adata, batch_key="n_section")
# mark the reference vs. query dataset
adata_concat.obs["is_ref"] = ["Query"] * len(test_embed_adata) + ["Reference"] * len(
    ref_embed_adata
)
adata_concat.obs["is_ref"] = adata_concat.obs["is_ref"].astype("category")
# mask the query dataset cell types
adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
adata_concat.obs[cell_type_key][: len(test_embed_adata)] = "To be predicted"
sc.pp.neighbors(adata_concat, use_rep="X_scGPT")
sc.tl.umap(adata_concat)
ref_cell_embeddings = ref_embed_adata.obsm["X_scGPT"]
test_emebd = test_embed_adata.obsm["X_scGPT"]
sc.settings.figdir =  "/maiziezhou_lab2/yuling/scGPT/tutorials/zero-shot/"
sc.pp.neighbors(adata_concat, use_rep="X_scGPT")
sc.tl.umap(adata_concat)
############
import matplotlib.pyplot as plt

sc.set_figure_params(
    dpi=100,  
    dpi_save=300,  
    frameon=False,
    figsize=(8, 6),  
    fontsize=12
)
fig = sc.pl.umap(
    adata_concat, 
    color=["is_ref", "original_clusters"], 
    wspace=0.4, 
    frameon=False, 
    ncols=1,
    show=False,
    return_fig=True,
    size=50,  
    legend_fontsize=10,
    legend_loc='right margin'
)
fig.suptitle('scGPT Reference Mapping', fontsize=14, y=1.02)
fig.savefig(
    'scGPT_reference_mapping.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    facecolor='white',
    edgecolor='none'
)

plt.close(fig)
########################################
def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims

def get_similar_vectors(vector, ref, top_k=10):
        # sims = cos_sim(vector, ref)
        sims = l2_sim(vector, ref)
        
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return top_k_idx, sims[top_k_idx]
k = 10  # number of neighbors


index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
index.add(ref_cell_embeddings)

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
distances, labels = index.search(test_emebd, k)

idx_list=[i for i in range(test_emebd.shape[0])]
preds = []
sim_list = distances
for k in idx_list:
    if faiss_imported:
        idx = labels[k]
    else:
        idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, k)
    pred = ref_embed_adata.obs[cell_type_key][idx].value_counts()
    preds.append(pred.index[0])
gt = test_adata.obs[cell_type_key].to_numpy()
results_df = pd.DataFrame({
    'cell_id': test_adata.obs.index.values,  
    'ground_truth': gt,
    'predictions': preds
})
results_df['correct'] = results_df['ground_truth'] == results_df['predictions']
results_df.to_csv('/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/scGPT/HumanLymph_reference_mapping/predictions_results.csv', index=False)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
test_embed_adata.obs['predicted_cell_type'] = preds
sc.pp.neighbors(test_embed_adata, use_rep="X_scGPT")
sc.tl.umap(test_embed_adata)
all_cell_types = list(set(list(test_embed_adata.obs['predicted_cell_type'].unique()) + 
                           list(test_embed_adata.obs['original_clusters'].unique())))
all_cell_types.sort()  
n_types = len(all_cell_types)
if n_types <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_types]
else:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_extra = plt.cm.tab20b(np.linspace(0, 1, n_types - 20))
    colors = np.vstack([colors, colors_extra])
color_dict = dict(zip(all_cell_types, [mcolors.rgb2hex(c) for c in colors]))
test_embed_adata.write_h5ad('/maiziezhou_lab2/yuling/scGPT/tutorials/zero-shot/reference_mapping_ad.h5ad')

sc.set_figure_params(
    dpi=100,
    dpi_save=300,
    frameon=False,
    figsize=(16, 6),  
    fontsize=12
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.umap(
    test_embed_adata, 
    color="predicted_cell_type",
    palette=color_dict,  
    ax=axes[0],
    show=False,
    size=50,
    legend_fontsize=9,
    legend_loc='right margin',
    title='Predicted',
    frameon=False
)
sc.pl.umap(
    test_embed_adata, 
    color="original_clusters",
    palette=color_dict,  
    ax=axes[1],
    show=False,
    size=50,
    legend_fontsize=9,
    legend_loc='right margin',
    title='Ground Truth',
    frameon=False
)

fig.suptitle('scGPT Cell Type Prediction (Query Only)', fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig(
    '/maiziezhou_lab2/yuling/scGPT/tutorials/zero-shot/scGPT_query_predictions_unified_colors.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
    facecolor='white',
    edgecolor='none'
)

plt.close(fig)
pd.DataFrame(list(color_dict.items()), columns=['cell_type', 'color']).to_csv(
    '/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/scGPT/HumanLymph_reference_mapping/color_mapping.csv',
    index=False
)

print("Query prediction figure with unified colors saved!")
print(f"Accuracy: {accuracy_score(gt, preds):.4f}")
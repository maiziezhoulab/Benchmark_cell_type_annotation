# DestVI

**DestVI** (Deconvolution of Spatial Transcriptomics profiles using Variational Inference)  
is a conditional generative model for spatial transcriptomics that captures **sub-cell-type variation**, enabling exploration of tissue spatial organization and gene expression heterogeneity across conditions.

### Advantages
- Stratifies cells into discrete cell types while modeling continuous sub-cell-type variation.  
- Scales to very large datasets (>1 million cells).  

### Limitations
-   Effectively requires a GPU for fast inference.

 
---

## Tasks

### 1. Cell Type Deconvolution
```python
proportions = st_model.get_proportions()
st_adata.obsm["proportions"] = proportions
```
Normalize to get spot-wise cell type fractions:

$$
\mathrm{Proportion}_{sc} = \frac{\beta_{sc}}{\sum_c \beta_{sc}}
$$

Plot:
```python
import scanpy as sc
st_adata.obs['B cells'] = st_adata.obsm['proportions']['B cells']
sc.pl.spatial(st_adata, color="B cells", spot_size=130)
```

### 2. Intra-cell-type Variation
```python
gamma = st_model.get_gamma()["B cells"]
st_adata.obsm["B_cells_gamma"] = gamma
```

### 3. Cell-Type-Specific Gene Expression Imputation
```python
indices = np.where(st_adata.obsm["proportions"][ct_name].values > 0.03)[0]
imputed_counts = st_model.get_scale_for_ct("Monocyte", indices=indices)[["Cxcl9", "Cxcl10", "Fcgr1"]]
```

### 4. Comparative Analysis Between Samples
Perform differential expression via frequentist tests using sampled parameters from the generative distribution.

### 5. Utility Functions
See [destvi_utils](https://destvi-utils.readthedocs.io/en/latest/installation.html) for:  
- Automatic thresholding of proportions  
- Spatial PCA analysis  
- Differential expression utilities  

---

## References

1. Romain Lopez *et al.* (2022). **DestVI identifies continuums of cell types in spatial transcriptomics data.** *Nature Biotechnology* (in press). [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.05.10.443517v1)  
2. Tomczak *et al.* (2018). *VAE with a VampPrior.* *AISTATS 2018.*  
3. Risso *et al.* (2018). *ZINB-WaVE: A general and flexible method for signal extraction from single-cell RNA-seq data.* *PNAS.*

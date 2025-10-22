# DestVI

**DestVI** (Deconvolution of Spatial Transcriptomics profiles using Variational Inference)  
is a conditional generative model for spatial transcriptomics that captures **sub-cell-type variation**, enabling exploration of tissue spatial organization and gene expression heterogeneity across conditions.

### Advantages
- Stratifies cells into discrete cell types while modeling continuous sub-cell-type variation.  
- Scales to very large datasets (>1 million cells).  

---

## Preliminaries

DestVI trains **two models**:
1. **scLVM** – single-cell latent variable model  
2. **stLVM** – spatial transcriptomics latent variable model  

- **scLVM** takes a scRNA-seq UMI count matrix \( X \in \mathbb{R}^{N \times G} \) and a vector of cell type labels \( \vec{c} \).  
- **stLVM** uses the trained scLVM together with a spatial transcriptomic matrix \( Y \in \mathbb{R}^{S \times G} \).  
- Optionally, the number of mixture components \( K \) for the empirical prior can be set.

---

## Generative Process

### scLVM

For each cell \( n \):
\[
\begin{align}
\gamma_n &\sim \mathcal{N}(0, I) \\
x_{ng} &\sim \mathrm{NegativeBinomial}(l_n f^g(c_n, \gamma_n), p_g)
\end{align}
\]

where:  
- \( l_n \): library size  
- \( f \): two-layer neural network decoder  
- \( p_g \): rate parameter of the Negative Binomial for gene \( g \)

> **Note**  
> We use the *rate–shape* parameterization of the Negative Binomial, enabling additive properties across cell types.

#### Latent Variables

| Latent variable | Description | Code variable |
|------------------|-------------|----------------|
| \( \gamma_n \in \mathbb{R}^d \) | Low-dimensional sub-cell-type covariate representation | `z` |
| \( p_g \in (0, \infty) \) | Rate parameter of the Negative Binomial | `px_r` |

---

### stLVM

Each spot \( s \) contains a mixture of cell types with abundances \( \beta_{sc} \).  
For spot \( s \) and gene \( g \):

\[
\begin{align}
\gamma_x^c &\sim \sum_{k=1}^K m_{kc} q_\Phi(\gamma^c \mid u_{kc}, c) \\
x_{sg} &\sim \mathrm{NegativeBinomial}\!\left(l_s \alpha_g \sum_{c=1}^{C} \beta_{sc} f^g(c, \gamma_s^c), p_g\right)
\end{align}
\]

where:  
- \( l_s \): library size  
- \( \alpha_g \): assay correction term  
- \( f \): decoder neural network  
- \( p_g \): rate parameter  

The empirical prior over \( \gamma_s^c \) is informed by scLVM subclusters \( u_{kc} \) for each cell type, forming a **VampPrior** (variational aggregated mixture of posteriors).

To capture gene-specific noise, a dummy “cell type” is introduced with  
\( \eta_g \sim \mathcal{N}(0,1) \) and \( \epsilon_g = \mathrm{Softplus}(\eta_g) \).

> Optional L1 regularization on \( \beta_{sc} \) increases sparsity of cell type proportions.

#### Latent Variables

| Latent variable | Description | Code variable |
|------------------|-------------|----------------|
| \( \beta_{sc} \in (0,\infty) \) | Spot-specific cell type abundance | `v_ind` |
| \( \gamma_s^c \in (-\infty,\infty) \) | Spot-specific sub-cell-type covariates | `gamma` |
| \( \eta_g \in (0,\infty) \) | Gene-specific noise | `eta` |
| \( \alpha_g \in (0,\infty) \) | Correction for technical differences | `beta` |
| \( p_g \in (0,\infty) \) | Rate parameter | `px_o` |

---

## Inference

### scLVM
- Uses **variational inference** via Auto-Encoding Variational Bayes.  
- Learns both model parameters (NN weights, rate parameters) and approximate posterior distributions.  
- Encoder implemented as `scvi.nn.Encoder`.

### stLVM
- Infers point estimates for \( \gamma^c, \alpha, \beta \) via penalized MAP.  
- Regularizes \( \alpha \) with a variance penalty and supports amortized inference for \( \beta \) and \( \gamma^c \).

The loss:

\[
\begin{align}
L(l,\alpha,\beta,f^g,\gamma,p,\eta) =& -\log p(X \mid l,\alpha,\beta,f^g,\gamma,p,\eta) - \lambda_\eta \log p(\eta) \\
&+ \lambda_\alpha \mathrm{Var}(\alpha) - \log p(\gamma \mid \text{VampPrior}) + \lambda_\beta \|\beta_{sc}\|_1
\end{align}
\]

Hyperparameters:
- \( \lambda_\beta \) (`l1_reg`): controls sparsity of proportions  
- \( \lambda_\alpha \) (`beta_reg`): regularizes assay correction  
- \( \lambda_\eta \) (`eta_reg`): controls dummy cell type contribution  

---

## Tasks

### 1. Cell Type Deconvolution
```python
proportions = st_model.get_proportions()
st_adata.obsm["proportions"] = proportions
```

Normalize to get spot-wise cell type fractions:
\[
\text{Proportion}_{sc} = \frac{\beta_{sc}}{\sum_c \beta_{sc}}
\]

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

library(reticulate)
library(Seurat)
library(anndata)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)
#anndata <- import("anndata")
sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/obj_integrated_sc_nucleus.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
unique_Section <- unique(seurat_obj@meta.data$n_section)
# load single cell data 
per_section_ref <- seurat_obj
counts <- per_section_ref@assays$originalexp@counts
cell_metadata <- as.data.frame(per_section_ref@meta.data)
RNAseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
RNAseurat_norm <- NormalizeData(RNAseurat, normalization.method = "LogNormalize", scale.factor = 10000)
RNAseurat_norm  <- ScaleData(RNAseurat_norm , assay = "RNA")
########### load spatial transcriptomics 
sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
per_section <- subset(seurat_obj, 
                        subset = Section.ID == '0503_F4_C')
counts <- per_section@assays$originalexp@counts
#counts <- as.matrix(counts)
# Extract cell metadata
#cell_metadata <- as.data.frame(ST$obs)
cell_metadata <- as.data.frame(per_section@meta.data)
STseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
STseurat_norm <- NormalizeData(STseurat, normalization.method = "LogNormalize", scale.factor = 10000)
STseurat_norm  <- ScaleData(STseurat_norm , assay = "RNA")
# Convert Seurat objects to SingleCellExperiment for SingleR
ref_data <- as.SingleCellExperiment(RNAseurat_norm, assay = "RNA")
merfish_data <- as.SingleCellExperiment(STseurat_norm, assay = "RNA")
cell_type_column <- "final_cluster_assignment" 
# Extract cell type labels
if (cell_type_column %in% colnames(RNAseurat_norm@meta.data)) {
  cell_types <- RNAseurat_norm@meta.data[[cell_type_column]]
  print(paste("Using cell type column:", cell_type_column))
  print(paste("Number of unique cell types:", length(unique(cell_types))))
  print("Cell type distribution:")
  print(table(cell_types))
} else {
  stop("Cell type annotation column not found. Please check your metadata column names.")
}
print("Running SingleR...")
pred_clust <- SingleR(
  test = merfish_data,
  ref = ref_data,
  labels = cell_types,
  de.method = "wilcox"
)

# Display results summary
print("SingleR prediction summary:")
print(table(pred_clust$labels))
STseurat_norm$SingleR_labels <- pred_clust$labels
STseurat_norm$SingleR_scores <- pred_clust$scores

print("SingleR analysis completed successfully!")
write.csv(pred_clust, file = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/scNucleus_output/singler_scNucleus.csv", row.names = TRUE)

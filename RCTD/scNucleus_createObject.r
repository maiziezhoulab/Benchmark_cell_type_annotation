library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
# conda env : conda activate r-gis

library(zellkonverter)
library(SingleCellExperiment)
library(readxl)
library(stringr)
library(Seurat)
 
library(zellkonverter)

sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/obj_integrated_sc_nucleus.h5ad")

# --- 1) Make cell & gene names unique
colnames(sce) <- make.unique(as.character(colnames(sce)))
rownames(sce) <- make.unique(as.character(rownames(sce)))

# --- 2) Sanitize gene symbols
rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- rownames(sce)
gene_sanitized <- gsub("\\|", "-", gene_sanitized)
gene_sanitized <- gsub("_",  "-", gene_sanitized)
gene_sanitized <- make.unique(gene_sanitized)
rownames(sce) <- gene_sanitized

if (!"counts" %in% assayNames(sce)) {
  assay(sce, "counts") <- assay(sce, "X")
}

stopifnot(all(c("counts", "X") %in% assayNames(sce)))
g_counts <- rownames(assay(sce, "counts"))
g_data   <- rownames(assay(sce, "X"))
common   <- intersect(g_counts, g_data)

sce <- sce[common, ]

seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
print(Assays(seurat_obj))
# --- 5) Quick checks
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]
DefaultAssay(seurat_obj) <- 'RNA'
 
##########################-----------------------------------------------------------------------------------------------------------------


library(spacexr)
library(Matrix)
##################################### csv to rds of ref data
### Load in/preprocess your data, this might vary based on your file type
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_ref"
# load counts file which saved from adata.X
counts <- read.csv(file.path(refdir, "counts.csv"), 
                header = TRUE, 
                row.names = 1, 
                colClasses = "character", 
                stringsAsFactors = FALSE)

counts_t <- t(counts)
# Step 3: Convert all values to integer
storage.mode(counts_t) <- "integer"  # works on matrix only
# Now you have: counts as integer matrix, row/col names preserved
counts <- counts_t
# counts <- t(read.csv(file.path(refdir, "counts.csv"), row.names = 1, check.names = FALSE)) # load in counts matrix
# meta_data <- read.csv(file.path(refdir,"meta_data.csv"), row.names = 1, check.names = FALSE) # load in meta_data (barcodes, clusters, and nUMI)

# load meta data of reference single cell 
meta_data <- read.table(file.path(refdir, "meta_data.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

cell_types <- meta_data$final_cluster_assignment
names(cell_types) <- rownames(meta_data) # create cell_types named list
cell_types <- as.factor(cell_types) # convert to factor data type

# nUMI <- as.numeric(meta_data$nUMI)
# names(nUMI) <- rownames(meta_data)

### Create the Reference object
levels(cell_types) <- gsub("/", "_", levels(cell_types))
reference <- Reference(counts, cell_types, min_UMI=20)
saveRDS(reference, file.path(refdir,'SCRef.rds'))
#----------------------------------------------------------------------
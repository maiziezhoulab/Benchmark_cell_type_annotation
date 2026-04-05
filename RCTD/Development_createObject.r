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
sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/Development.h5ad")
##############
library(SingleCellExperiment)
library(zellkonverter)
library(Seurat)

# --- 1) Make cell & gene names unique
colnames(sce) <- make.unique(as.character(colnames(sce)))
rownames(sce) <- make.unique(as.character(rownames(sce)))

# --- 2) (Recommended) sanitize gene symbols so Seurat won’t rewrite them.
#     Keep a mapping so you can get back the originals if needed.
rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- rownames(sce)
gene_sanitized <- gsub("\\|", "-", gene_sanitized)
gene_sanitized <- gsub("_",  "-", gene_sanitized)
gene_sanitized <- make.unique(gene_sanitized)
rownames(sce) <- gene_sanitized

# --- 3) Ensure both assays have exactly the same features (names + order)
stopifnot(all(c("counts", "X") %in% assayNames(sce)))
g_counts <- rownames(assay(sce, "counts"))
g_data   <- rownames(assay(sce, "X"))
common   <- intersect(g_counts, g_data)

# subset AND align order
sce <- sce[common, ]
# (after subsetting, both assays share the same rownames & order)

# --- 4) Convert (counts = "counts", data = "X")
seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")

# --- 5) Quick checks
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]
DefaultAssay(seurat_obj) <- 'RNA'
 
##########################-----------------------------------------------------------------------------------------------------------------
#seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
rd_name <-  "spatial"
coords <- as.data.frame(reducedDim(sce, rd_name))
colnames(coords)[1:2] <- c("center_x", "center_y")
coords <- coords[colnames(seurat_obj), c("center_x","center_y"), drop = FALSE]
seurat_obj <- AddMetaData(seurat_obj, metadata = coords)

##### reference 
total_data <- subset(seurat_obj, 
                        subset = Batch == 'Stage54_telencephalon_rep2_DP8400015649BRD6_2')
# query data 
per_section_spatial <- subset(seurat_obj, 
                        subset = Batch == 'Stage44_telencephalon_rep2_FP200000239BL_E4')
spatial_count <- per_section_spatial@assays$originalexp$data
library(spacexr)
library(Matrix)
##################################### csv to rds of ref data
### Load in/preprocess your data, this might vary based on your file type
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_ref"
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

cell_types <- meta_data$Annotation
names(cell_types) <- rownames(meta_data) # create cell_types named list
cell_types <- as.factor(cell_types) # convert to factor data type

# nUMI <- as.numeric(meta_data$nUMI)
# names(nUMI) <- rownames(meta_data)

### Create the Reference object
#levels(cell_types) <- gsub("/", "_", levels(cell_types))
reference <- Reference(counts, cell_types, min_UMI=20)
saveRDS(reference, file.path(refdir,'SCRef.rds'))
#------------------------------------------------------------------------------
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_input"
# Step 1: Read everything as character to avoid numeric coercion
raw <- read.csv(file.path(datadir, "counts.csv"), 
                header = TRUE, 
                row.names = 1, 
                colClasses = "character", 
                stringsAsFactors = FALSE)
# Step 2: Transpose (genes as rows, barcodes as columns, or vice versa)
raw_t <- t(raw)
# Step 3: Convert all values to integer
storage.mode(raw_t) <- "integer"  # works on matrix only
# Now you have: counts as integer matrix, row/col names preserved
counts <- raw_t

# Step 1: Read all as character to preserve long barcodes
coords_raw <- read.table(file.path(datadir, "location.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

coords <- data.frame(lapply(coords_raw, as.numeric), row.names = rownames(coords_raw))

# rownames(coords) <- coords$barcodes; coords$barcodes <- NULL # Move barcodes to rownames
# nUMI <- colSums(counts) # In this case, total counts per pixel is nUMI

### Create SpatialRNA object
mSC <- SpatialRNA(coords, counts)
saveRDS(mSC, file.path(datadir,'SCRaw.rds'))



library(Seurat)
library(Matrix)

refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_ref"
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

meta_data <- read.table(file.path(refdir, "meta_data.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

ref <- CreateSeuratObject(counts=counts, meta.data = meta_data)

############################################ Load spatial transcriptomics data 
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_input"

counts <- read.csv(file.path(datadir, "counts.csv"), 
                header = TRUE, 
                row.names = 1, 
                colClasses = "character", 
                stringsAsFactors = FALSE)

counts_t <- t(counts)
# Step 3: Convert all values to integer
storage.mode(counts_t) <- "integer"  # works on matrix only
# Now you have: counts as integer matrix, row/col names preserved
counts <- counts_t

meta_data <- read.table(file.path(datadir, "meta_data.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

raw <- CreateSeuratObject(counts=counts, meta.data = meta_data)

############################################

# Preprocess reference
ref <- NormalizeData(ref)
ref <- FindVariableFeatures(ref)
ref <- ScaleData(ref)
ref <- RunPCA(ref)

# Preprocess query
raw <- NormalizeData(raw)
raw <- FindVariableFeatures(raw)
raw <- ScaleData(raw)
raw <- RunPCA(raw)

##############################

# Find anchors
anchors <- FindTransferAnchors(
  reference = ref,
  query = raw,
  dims = 1:30,
  reference.reduction = "pca"
)

# Transfer labels (assuming ref$celltype has annotations)
predictions <- TransferData(
  anchorset = anchors,
  refdata = ref@meta.data$final_cluster_assignment,
  dims = 1:30
)
# Add predictions to query metadata
# raw <- AddMetaData(raw, metadata = predictions)
write.csv(predictions, file = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/scNucleus_output/seurat_Nucleus.csv")

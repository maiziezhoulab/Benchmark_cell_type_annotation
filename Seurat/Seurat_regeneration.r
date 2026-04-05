library(Seurat)
library(Matrix)
# conda activate r-gis
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/regeneration_ref"
library(peakRAM)
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
#########################################################
meta_data <- read.table(file.path(refdir, "meta_data.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

ref <- CreateSeuratObject(counts=counts, meta.data = meta_data)
############################################ Load spatial transcriptomics data 
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/regeneration_input"

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
###################################################################
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
res_peak <- peakRAM({
  anchors <- FindTransferAnchors(
    reference = ref,
    query = raw,
    dims = 1:30,
    reference.reduction = "pca"
  )

  predictions <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data$Annotation,
    dims = 1:30
  )
})
    elapsed_sec <- if ("Elapsed_Time_sec" %in% colnames(res_peak)) {
    as.numeric(res_peak$Elapsed_Time_sec[1])
  } else if ("Elapsed_Time_s" %in% colnames(res_peak)) {
    as.numeric(res_peak$Elapsed_Time_s[1])
  } else {
    NA_real_
  }
###################################################################################################################
    peak_ram_mb <- if ("Peak_RAM_Used_MiB" %in% colnames(res_peak)) {
    as.numeric(res_peak$Peak_RAM_Used_MiB[1])
  } else if ("Peak_RAM_Used_Mb" %in% colnames(res_peak)) {
    as.numeric(res_peak$Peak_RAM_Used_Mb[1])
  } else {
    NA_real_
  }
# raw <- AddMetaData(raw, metadata = predictions)
write.csv(predictions, file = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/regeneration_output/seurat_development.csv")
runtime_all <- data.frame(
      Time_sec = elapsed_sec,
      Peak_Memory = peak_ram_mb
    )

write.csv(
  runtime_all,
  file.path("/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/regeneration_output", "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)
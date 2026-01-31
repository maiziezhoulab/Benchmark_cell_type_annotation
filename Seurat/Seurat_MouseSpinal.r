library(Seurat)
library(Matrix)
library(peakRAM)   # peak memory
library(tictoc)    # runtime

# -----------------------
# Paths
# -----------------------
datadir_raw <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/ref_0503_F4_C"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/0503_F4_C_outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# -----------------------
# Parameters
# -----------------------
k <- 17
dims_use <- 1:30

# -----------------------
# Helper: create Seurat object
# -----------------------
make_seurat_from_counts_meta <- function(dir_path) {

  counts_df <- read.csv(
    file.path(dir_path, "counts.csv"),
    header = TRUE,
    row.names = 1,
    colClasses = "character",
    stringsAsFactors = FALSE
  )
  counts_mat <- t(as.matrix(counts_df))
  storage.mode(counts_mat) <- "integer"

  meta_data <- read.table(
    file.path(dir_path, "meta_data.csv"),
    header = TRUE,
    row.names = 1,
    sep = ",",
    colClasses = "character",
    stringsAsFactors = FALSE
  )

  obj <- CreateSeuratObject(
    counts = counts_mat,
    meta.data = meta_data
  )

  return(obj)
}

# =======================
# Load & preprocess RAW (once)
# =======================
raw <- make_seurat_from_counts_meta(datadir_raw)

raw <- NormalizeData(raw)
raw <- FindVariableFeatures(raw)
raw <- ScaleData(raw)
raw <- RunPCA(raw)

# =======================
# Load reference (series = 17)
# =======================
series_dir <- file.path(series_base, paste0("series_", k))
stopifnot(dir.exists(series_dir))

message(">>> Running Seurat label transfer for series_", k)

ref <- make_seurat_from_counts_meta(series_dir)

if (!"MERFISH.cell.type.annotation" %in% colnames(ref@meta.data)) {
  stop("meta_data lacks MERFISH.cell.type.annotation: ", series_dir)
}

ref <- NormalizeData(ref)
ref <- FindVariableFeatures(ref)
ref <- ScaleData(ref)
ref <- RunPCA(ref)

# =======================
# START PROFILING
# =======================
gc()

peak_res <- peakRAM({

  anchors <- FindTransferAnchors(
    reference = ref,
    query = raw,
    dims = dims_use,
    reference.reduction = "pca"
  )

  predictions <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data$MERFISH.cell.type.annotation,
    dims = dims_use
  )
})
runtime_sec  <- peak_res$Elapsed_Time_sec
peak_mem_MiB <- peak_res$Peak_RAM_Used_MiB

# =======================
# Save outputs
# =======================
outfile_pred <- file.path(outdir, paste0("seurat_series_", k, "_pred.csv"))
write.csv(predictions, outfile_pred)

profile_df <- data.frame(
  runtime_sec = runtime_sec,
  peak_memory_MiB = peak_mem_MiB
)

outfile_profile <- file.path(outdir, "runtimeSec_memoryMiB.csv")
write.csv(profile_df, outfile_profile, row.names = FALSE)

message("=== DONE ===")
print(profile_df)

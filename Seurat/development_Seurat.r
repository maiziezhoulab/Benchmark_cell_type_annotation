############################################
## Seurat label transfer benchmark
## Time + peak memory via peakRAM
############################################

library(Seurat)
library(Matrix)
library(peakRAM)

## -------------------------------
## Paths
## -------------------------------
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_ref"
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_input"
outdir  <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Development_output"

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

############################################
## Load reference data
############################################
counts_ref <- read.csv(
  file.path(refdir, "counts.csv"),
  header = TRUE,
  row.names = 1,
  colClasses = "character",
  stringsAsFactors = FALSE
)

counts_ref <- t(as.matrix(counts_ref))
storage.mode(counts_ref) <- "integer"

meta_ref <- read.table(
  file.path(refdir, "meta_data.csv"),
  header = TRUE,
  row.names = 1,
  sep = ",",
  colClasses = "character",
  stringsAsFactors = FALSE
)

ref <- CreateSeuratObject(
  counts = counts_ref,
  meta.data = meta_ref
)

############################################
## Load spatial (query) data
############################################
counts_q <- read.csv(
  file.path(datadir, "counts.csv"),
  header = TRUE,
  row.names = 1,
  colClasses = "character",
  stringsAsFactors = FALSE
)

counts_q <- t(as.matrix(counts_q))
storage.mode(counts_q) <- "integer"

meta_q <- read.table(
  file.path(datadir, "meta_data.csv"),
  header = TRUE,
  row.names = 1,
  sep = ",",
  colClasses = "character",
  stringsAsFactors = FALSE
)

raw <- CreateSeuratObject(
  counts = counts_q,
  meta.data = meta_q
)

############################################
## Preprocessing (NOT benchmarked)
############################################
ref <- NormalizeData(ref)
ref <- FindVariableFeatures(ref)
ref <- ScaleData(ref)
ref <- RunPCA(ref)

raw <- NormalizeData(raw)
raw <- FindVariableFeatures(raw)
raw <- ScaleData(raw)
raw <- RunPCA(raw)

############################################
## Benchmark Seurat label transfer
############################################
res <- peakRAM({

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

############################################
## Extract runtime & memory
############################################
runtime_sec   <- res$Elapsed_Time_sec
peak_mem_MiB  <- res$Peak_RAM_Used_MiB

############################################
## Save benchmark results
############################################
summary_df <- data.frame(
  Elapsed_Time_sec = runtime_sec,
  Peak_RAM_Used_MiB = peak_mem_MiB
)

write.csv(
  summary_df,
  file.path(outdir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

############################################
## Save predictions
############################################
write.csv(
  predictions,
  file.path(outdir, "seurat_development.csv")
)

############################################
## Print summary
############################################
cat("Seurat label transfer benchmark finished\n")
cat("Elapsed time (sec): ", runtime_sec, "\n")
cat("Peak RAM used (MiB): ", peak_mem_MiB, "\n")

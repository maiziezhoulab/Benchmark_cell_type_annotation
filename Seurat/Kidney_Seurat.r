library(Seurat)
library(Matrix)
library(pryr)
library(dplyr)

## ---------------- Benchmark environment ----------------
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)

## ---------------- Parameters ----------------
query <- c("L", "R")
time_point <- c("Sham", "Hour4", "Hour12", "Day2", "Day14", "Week6")


datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_input"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Kidney_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

## ---------------- Load reference ONCE ----------------
counts_ref <- read.csv(
  "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_all_ref/counts.csv",
  header = TRUE,
  row.names = 1,
  colClasses = "character",
  stringsAsFactors = FALSE
)
counts_ref <- t(counts_ref)
storage.mode(counts_ref) <- "integer"

meta_ref <- read.table(
  "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_all_ref/meta_data.csv",
  header = TRUE,
  row.names = 1,
  sep = ",",
  colClasses = "character",
  stringsAsFactors = FALSE
)

ref <- CreateSeuratObject(counts = counts_ref, meta.data = meta_ref)

ref <- NormalizeData(ref)
ref <- FindVariableFeatures(ref)
ref <- ScaleData(ref)
ref <- RunPCA(ref)

## ---------------- Benchmark result container ----------------
perf_list <- list()

## ======================= Main loop =======================
for (t in time_point) {
  for (q in query) {

    message("Running Seurat LT: ", t, " ", q)

    ## ---- Load query data ----
    counts_q <- read.csv(
      file.path(datadir, paste0(t, q, "_counts.csv")),
      header = TRUE,
      row.names = 1,
      colClasses = "character",
      stringsAsFactors = FALSE
    )
    counts_q <- t(counts_q)
    storage.mode(counts_q) <- "integer"

    meta_q <- read.table(
      file.path(datadir, paste0(t, q, "meta_data.csv")),
      header = TRUE,
      row.names = 1,
      sep = ",",
      colClasses = "character",
      stringsAsFactors = FALSE
    )

    raw <- CreateSeuratObject(counts = counts_q, meta.data = meta_q)

    raw <- NormalizeData(raw)
    raw <- FindVariableFeatures(raw)
    raw <- ScaleData(raw)
    raw <- RunPCA(raw)

    ## ---- Benchmark: anchors + transfer ONLY ----
    gc()

    t_start <- Sys.time()

    mem_change <- pryr::mem_change({

      anchors <- FindTransferAnchors(
        reference = ref,
        query = raw,
        dims = 1:30,
        reference.reduction = "pca"
      )

      predictions <- TransferData(
        anchorset = anchors,
        refdata = ref@meta.data$celltype,
        dims = 1:30
      )
    })

    t_end <- Sys.time()

    time_sec <- as.numeric(difftime(t_end, t_start, units = "secs"))
    mem_mib  <- as.numeric(mem_change) / 1024^2

    ## ---- Save prediction (NOT timed) ----
    write.csv(
      predictions,
      file.path(outdir, paste0(t, q, "_Kidney.csv"))
    )

    ## ---- Collect benchmark result ----
    perf_list[[length(perf_list) + 1]] <- data.frame(
      method = "Seurat_LabelTransfer",
      dataset = "Xenium",
      time_point = t,
      section = q,
      n_spatial_cells = ncol(raw),
      Time_sec = time_sec,
      Memory_MiB = mem_mib
    )
  }
}

## ======================= Save benchmark table =======================
perf_all <- bind_rows(perf_list)

write.csv(
  perf_all,
  file.path(outdir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("Seurat benchmark finished.")

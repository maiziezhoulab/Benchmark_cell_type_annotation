library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(CARD)
library(zellkonverter)
library(SingleCellExperiment)
library(readxl)
library(stringr)
library(peakRAM)

time_point <- c('Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6')
perf_list <- list()
for (t in time_point) {
  sce <- readH5AD(paste0(
    "/maiziezhou_lab2/yuling/datasets/Kidney/snRNA-seq/time_", t, ".h5ad"
  ))
  colnames(sce) <- make.unique(as.character(colnames(sce)))
  rownames(sce) <- make.unique(as.character(rownames(sce)))
  seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
  seurat_obj <- FindVariableFeatures(
    seurat_obj,
    selection.method = "vst",
    nfeatures = 3000
  )
  seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]
  sc_count <- seurat_obj@assays$originalexp$data
  sc_meta  <- seurat_obj@meta.data
  sc_meta  <- dplyr::rename(sc_meta, cellType = name)
  sc_meta$sampleInfo <- "sample1"

  sections <- c("L", "R")
  for (i in sections) {
    ## ---- Spatial data ----
    sce_sp <- readH5AD(paste0(
      "/maiziezhou_lab2/yuling/datasets/Kidney/Xenium/time_", t, i, ".h5ad"
    ))
    per_section_spatial <- as.Seurat(sce_sp, counts = "counts", data = "X")
    spatial_count <- per_section_spatial@assays$originalexp$counts
    meta <- per_section_spatial[[]]
    centers <- meta[, c("x_centroid", "y_centroid"), drop = FALSE]
    spatial_location <- cbind(
      cell_id = rownames(centers),
      setNames(centers, c("x", "y"))
    )
    rownames(spatial_location) <- NULL
    spatial_location <- tibble::column_to_rownames(
      spatial_location, var = "cell_id"
    )

    ## ---- CARD object (preprocessing, NOT timed) ----
    CARD_obj <- createCARDObject(
      sc_count = sc_count,
      sc_meta = sc_meta,
      spatial_count = spatial_count,
      spatial_location = spatial_location,
      ct.varname = "cellType",
      ct.select = unique(sc_meta$cellType),
      sample.varname = "sampleInfo",
      minCountGene = 0,
      minCountSpot = 0
    )
    ## ---- MODEL ONLY: time + peak memory ----
    peak_res <- peakRAM::peakRAM({
      CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)
    })

    time_sec  <- peak_res$Elapsed_Time_sec[1]
    peak_mem <- peak_res$Peak_RAM_MiB[1]

    ## ---- Output ----
    out_dir <- paste0(
      "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/Kidney_output",
      t, "_", i
    )
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    write.csv(
      CARD_obj@Proportion_CARD,
      file.path(out_dir, "CARD_predictions.csv"),
      quote = FALSE
    )
    perf_list[[length(perf_list) + 1]] <- data.frame(
    method = "CARD",
    dataset = "Xenium",
    time_point = t,
    section = i,
    Time = time_sec,
    Peak_Memory_MiB = peak_mem
  )}
}
perf_all <- dplyr::bind_rows(perf_list)
  out_perf <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/Kidney_output/runtimeSec_memoryMiB.csv"
  write.csv(
    perf_all,
    out_perf,
    row.names = FALSE,
    quote = FALSE
  )
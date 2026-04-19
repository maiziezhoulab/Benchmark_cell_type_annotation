library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
# conda env : conda activate r-gis
#library(SeuratWrappers)
library(CARD)
#library(viridis)
library(zellkonverter)
library(SingleCellExperiment)
library(readxl)
library(stringr)

outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/scNucleus_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
metrics_csv <- file.path(outdir, "runtimeSec_memoryMiB.csv")

## Peak resident set (Linux): VmHWM in kB → MiB; NA if unavailable
read_vmhwm_mib <- function() {
  if (!file.exists("/proc/self/status")) {
    return(NA_real_)
  }
  st <- readLines("/proc/self/status", warn = FALSE)
  line <- grep("^VmHWM:", st, value = TRUE)
  if (!length(line)) {
    return(NA_real_)
  }
  parts <- strsplit(trimws(line[1]), "\\s+")[[1]]
  as.numeric(parts[2]) / 1024
}

peak_mem_gc_mib <- function() {
  g <- gc(full = TRUE)
  if (!is.null(colnames(g)) && "max used" %in% colnames(g)) {
    return(sum(as.numeric(g[, "max used"]), na.rm = TRUE))
  }
  sum(as.numeric(g[, ncol(g)]), na.rm = TRUE)
}

## Wall time: from first data load through writing outputs
t0 <- proc.time()

sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/obj_integrated_sc_nucleus.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X")
total_data <- seurat_obj
rd_name <- "spatial"
#########################################
sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X")
per_section_spatial <- subset(seurat_obj,
                              subset = Section.ID == "0503_F4_C")
spatial_count <- per_section_spatial@assays$originalexp$data
#----------- spatial location data must be in the format of data frame
meta <- per_section_spatial[[]]
centers <- meta[, c("center_x", "center_y"), drop = FALSE]
spatial_location <- cbind(cell_id = rownames(centers),
                            setNames(centers, c("x", "y")))
rownames(spatial_location) <- NULL
spatial_location <- tibble::column_to_rownames(spatial_location, var = "cell_id")
#----------------- single cell data
sc_count <- total_data@assays$originalexp$data
sc_meta <- total_data@meta.data
sc_meta <- dplyr::rename(sc_meta, cellType = final_cluster_assignment)
sc_meta$sampleInfo <- "sample1"

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
CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)

pred_path <- file.path(outdir, "CARD_predictions.csv")
write.csv(CARD_obj@Proportion_CARD, pred_path, quote = FALSE)

elapsed_sec <- as.numeric(proc.time() - t0)[3]
gc(full = TRUE)
peak_mib <- read_vmhwm_mib()
if (is.na(peak_mib)) {
  peak_mib <- peak_mem_gc_mib()
}

write.csv(
  data.frame(Time = elapsed_sec, Memory = peak_mib),
  file = metrics_csv,
  row.names = FALSE
)

message("Saved predictions: ", pred_path)
message("Saved metrics (Time = wall sec, Memory ≈ peak MiB): ", metrics_csv)

# conda activate cs336_basics
############################################
## SingleR benchmark: time + peak memory
## using peakRAM
############################################

library(reticulate)
library(Seurat)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)
library(peakRAM)

## -------------------------------
## Paths
## -------------------------------
h5ad_path <- "/maiziezhou_lab2/yuling/datasets/Development.h5ad"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/Development_output"

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

############################################
## Load data (NOT benchmarked)
############################################
sce <- readH5AD(h5ad_path)

## Make names unique
colnames(sce) <- make.unique(as.character(colnames(sce)))
rownames(sce) <- make.unique(as.character(rownames(sce)))

## Sanitize gene symbols for Seurat
rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- rownames(sce)
gene_sanitized <- gsub("\\|", "-", gene_sanitized)
gene_sanitized <- gsub("_", "-", gene_sanitized)
gene_sanitized <- make.unique(gene_sanitized)
rownames(sce) <- gene_sanitized

## Align assays
stopifnot(all(c("counts", "X") %in% assayNames(sce)))
common <- intersect(rownames(assay(sce, "counts")),
                    rownames(assay(sce, "X")))
sce <- sce[common, ]

############################################
## Convert to Seurat (NOT benchmarked)
############################################
seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")

############################################
## Reference (Stage54)
############################################
ref_seurat <- subset(
  seurat_obj,
  subset = Batch == "Stage54_telencephalon_rep2_DP8400015649BRD6_2"
)

ref_seurat <- CreateSeuratObject(
  counts = ref_seurat@assays$originalexp@counts,
  meta.data = as.data.frame(ref_seurat@meta.data)
)

ref_seurat <- NormalizeData(ref_seurat)
ref_seurat <- ScaleData(ref_seurat)

############################################
## Query (Stage44)
############################################
query_seurat <- subset(
  seurat_obj,
  subset = Batch == "Stage44_telencephalon_rep2_FP200000239BL_E4"
)

query_seurat <- CreateSeuratObject(
  counts = query_seurat@assays$originalexp@counts,
  meta.data = as.data.frame(query_seurat@meta.data)
)

query_seurat <- NormalizeData(query_seurat)
query_seurat <- ScaleData(query_seurat)

############################################
## Convert to SingleCellExperiment
############################################
ref_data <- as.SingleCellExperiment(ref_seurat, assay = "RNA")
test_data <- as.SingleCellExperiment(query_seurat, assay = "RNA")

cell_type_column <- "Annotation"

if (!cell_type_column %in% colnames(colData(ref_data))) {
  stop("Cell type annotation column not found in reference metadata.")
}

cell_types <- colData(ref_data)[[cell_type_column]]

############################################
## Benchmark SingleR
############################################
res <- peakRAM({

  pred_clust <- SingleR(
    test = test_data,
    ref = ref_data,
    labels = cell_types,
    de.method = "wilcox"
  )

})

############################################
## Extract runtime & memory
############################################
runtime_sec  <- res$Elapsed_Time_sec
peak_mem_MiB <- res$Peak_RAM_Used_MiB

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
  pred_clust,
  file.path(outdir, "Results.csv"),
  row.names = TRUE
)

############################################
## Print summary
############################################
cat("SingleR benchmark finished\n")
cat("Elapsed time (sec): ", runtime_sec, "\n")
cat("Peak RAM used (MiB): ", peak_mem_MiB, "\n")

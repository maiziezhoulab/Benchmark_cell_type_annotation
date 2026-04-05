library(reticulate)
library(Seurat)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)
library(peakRAM)
# conda activate cs336_basics
## -------------------------------
## Paths
## -------------------------------
h5ad_path <- "/maiziezhou_lab2/yuling/datasets/Develop/5DPIs.h5ad"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/regeneration_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
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
seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
############################################
## Reference (Stage54)
############################################
# ref_seurat <- subset(
#   seurat_obj,
#   subset = Batch == "Stage54_telencephalon_rep2_DP8400015649BRD6_2"
# )
ref_seurat <- subset(
  seurat_obj,
  subset = Batch %in% c(
    "Injury_5DPI_rep1_SS200000147BL_D2",
    "Injury_5DPI_rep2_SS200000147BL_D2"
  )
)
old_names <- Cells(ref_seurat)
batch_vec  <- ref_seurat$Batch
new_names <- paste0(old_names, "_", batch_vec)
any(duplicated(new_names))
ref_seurat <- RenameCells(ref_seurat, new.names = new_names)
ref_seurat <- CreateSeuratObject(
  counts = ref_seurat@assays$originalexp@counts,
  meta.data = as.data.frame(ref_seurat@meta.data)
)
ref_seurat <- NormalizeData(ref_seurat)
ref_seurat <- ScaleData(ref_seurat)
############################
query_seurat <- subset(
  seurat_obj,
  subset = Batch == "Injury_5DPI_rep3_SS200000147BL_D3"
)
query_seurat <- CreateSeuratObject(
  counts = query_seurat@assays$originalexp@counts,
  meta.data = as.data.frame(query_seurat@meta.data)
)

query_seurat <- NormalizeData(query_seurat)
query_seurat <- ScaleData(query_seurat)
ref_data <- as.SingleCellExperiment(ref_seurat, assay = "RNA")
test_data <- as.SingleCellExperiment(query_seurat, assay = "RNA")

cell_type_column <- "Annotation"

if (!cell_type_column %in% colnames(colData(ref_data))) {
  stop("Cell type annotation column not found in reference metadata.")
}

cell_types <- colData(ref_data)[[cell_type_column]]
############################################################################
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
  Time_sec = runtime_sec,
  Peak_Memory = peak_mem_MiB
)
write.csv(
  summary_df,
  file.path(outdir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)
write.csv(
  pred_clust,
  file.path(outdir, "Results.csv"),
  row.names = TRUE
)
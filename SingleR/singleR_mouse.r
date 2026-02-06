# ============================================================
# SingleR (k=17 only) + benchmark core runtime & peak memory
# Excluding preprocessing time (data loading / normalization)
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleR)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(peakRAM)
})

# --------------------- paths ---------------------
in_h5ad <- "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
base_out <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/0503_F4_C_output_k17"
dir.create(base_out, showWarnings = FALSE, recursive = TRUE)

pred_csv <- file.path(base_out, "singler_predictions.csv")
bench_csv <- file.path(base_out, "runtimeSec_memoryMiB.csv")

# --------------------- params ---------------------
target_section <- "0503_F4_C"
prefix_keep <- "^0503"
remove_sections <- c("0503_nan_nan", "0503_F4_C")
k_target <- 17
cell_type_column <- "MERFISH.cell.type.annotation"

# --------------------- helper ---------------------
get_assay_counts <- function(obj) {
  # Prefer originalexp@counts if exists, else RNA@counts
  if ("originalexp" %in% names(obj@assays)) {
    return(GetAssayData(obj, assay = "originalexp", slot = "counts"))
  } else if ("RNA" %in% names(obj@assays)) {
    return(GetAssayData(obj, assay = "RNA", slot = "counts"))
  } else {
    stop("No suitable assay found (expected 'originalexp' or 'RNA').")
  }
}

safe_celltype_col <- function(meta_df, preferred = "MERFISH.cell.type.annotation") {
  if (preferred %in% colnames(meta_df)) return(preferred)
  candidates <- c("cell_type", "celltype", "Cell_Type", "annotation",
                  "MERFISH_cell_type_annotation", "cluster", "seurat_clusters")
  hit <- candidates[candidates %in% colnames(meta_df)]
  if (length(hit) > 0) return(hit[1])
  stop("Cell type annotation column not found in metadata.")
}

# --------------------- load once (preprocessing, NOT counted) ---------------------
sce <- readH5AD(in_h5ad)
seurat_obj <- as.Seurat(sce, counts = "X", data = "X")

unique_sections <- unique(seurat_obj@meta.data$Section.ID)
selected <- grep(prefix_keep, unique_sections, value = TRUE)
selected <- setdiff(selected, remove_sections)

if (length(selected) == 0) stop("No reference sections after filtering.")
k <- min(k_target, length(selected))
ids_k <- selected[seq_len(k)]

message("Using k = ", k)
message("Reference sections:")
print(ids_k)
message("Target section: ", target_section)

# Build reference Seurat (preprocessing, NOT counted)
ref_seu <- subset(seurat_obj, subset = Section.ID %in% ids_k)
ref_counts <- get_assay_counts(ref_seu)
ref_meta <- as.data.frame(ref_seu@meta.data)

RNAseurat <- CreateSeuratObject(counts = ref_counts, meta.data = ref_meta)
RNAseurat <- NormalizeData(RNAseurat, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
RNAseurat <- ScaleData(RNAseurat, assay = "RNA", verbose = FALSE)

# Build target Seurat (preprocessing, NOT counted)
test_seu <- subset(seurat_obj, subset = Section.ID == target_section)
if (ncol(test_seu) == 0) stop("Target section has 0 cells/spots: ", target_section)

test_counts <- get_assay_counts(test_seu)
test_meta <- as.data.frame(test_seu@meta.data)

STseurat <- CreateSeuratObject(counts = test_counts, meta.data = test_meta)
STseurat <- NormalizeData(STseurat, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
STseurat <- ScaleData(STseurat, assay = "RNA", verbose = FALSE)

# Convert to SCE (preprocessing, NOT counted)
ref_data <- as.SingleCellExperiment(RNAseurat, assay = "RNA")
test_data <- as.SingleCellExperiment(STseurat, assay = "RNA")

# cell labels
cell_type_column <- safe_celltype_col(RNAseurat@meta.data, preferred = cell_type_column)
cell_types <- RNAseurat@meta.data[[cell_type_column]]
message("Using cell type column: ", cell_type_column)
message("Unique cell types: ", length(unique(cell_types)))

# --------------------- CORE BENCHMARK (counted) ---------------------
ram_time <- peakRAM({
  pred_clust <- SingleR(
    test = test_data,
    ref = ref_data,
    labels = cell_types,
    de.method = "wilcox"
  )
})

# --------------------- save outputs (NOT counted) ---------------------
pred_df <- as.data.frame(pred_clust)
write.csv(pred_df, pred_csv, row.names = TRUE)

# SingleR output label summary
label_tab <- sort(table(pred_clust$labels), decreasing = TRUE)
print(label_tab)

# benchmark table
benchmark_df <- data.frame(
  method = "SingleR",
  k = k,
  target_section = target_section,
  n_ref_sections = length(ids_k),
  n_ref_cells = ncol(ref_data),
  n_test_cells = ncol(test_data),
  runtime_sec = as.numeric(ram_time$Elapsed_Time_sec[1]),
  peak_memory_MiB = as.numeric(ram_time$Peak_RAM_Used_MiB[1]),
  stringsAsFactors = FALSE
)

write.csv(benchmark_df, bench_csv, row.names = FALSE)
print(benchmark_df)

message("Done.")
message("Predictions saved to: ", pred_csv)
message("Benchmark saved to: ", bench_csv)

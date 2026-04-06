library(reticulate)
library(Seurat)
library(anndata)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)
library(peakRAM)     # for peak memory

# load scRNA-seq reference
sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
unique_Section <- unique(seurat_obj@meta.data$Section.ID)
selected_0503 <- grep("^0503", unique_Section, value = TRUE)
selected_0503_clean <- setdiff(selected_0503, "0503_nan_nan")
selected_0503_1 <- setdiff(selected_0503_clean, "0503_F4_C")
# output dir
base_out <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/0503_F4_C_output"
dir.create(base_out, showWarnings = FALSE, recursive = TRUE)

# results table
perf_df <- data.frame(
  series = integer(),
  time_sec = numeric(),
  peak_mem_MB = numeric(),
  stringsAsFactors = FALSE
)

# loop
for (k in 1:17) {
  cat("Running series", k, "...\n")

  start_time <- Sys.time()

  ram_res <- peakRAM({

    ids_k <- selected_0503_1[seq_len(k)]   # first k elements
    per_section_ref <- subset(
      seurat_obj,
      subset = Section.ID %in% ids_k
    )

    counts <- per_section_ref@assays$originalexp@counts
    cell_metadata <- as.data.frame(per_section_ref@meta.data)

    RNAseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
    RNAseurat_norm <- NormalizeData(
      RNAseurat,
      normalization.method = "LogNormalize",
      scale.factor = 10000
    )
    RNAseurat_norm <- ScaleData(RNAseurat_norm, assay = "RNA")

    per_section <- subset(
      seurat_obj,
      subset = Section.ID == "0503_F4_C"
    )

    counts <- per_section@assays$originalexp@counts
    cell_metadata <- as.data.frame(per_section@meta.data)

    STseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
    STseurat_norm <- NormalizeData(
      STseurat,
      normalization.method = "LogNormalize",
      scale.factor = 10000
    )
    STseurat_norm <- ScaleData(STseurat_norm, assay = "RNA")

    # Convert Seurat objects to SingleCellExperiment
    ref_data <- as.SingleCellExperiment(RNAseurat_norm, assay = "RNA")
    merfish_data <- as.SingleCellExperiment(STseurat_norm, assay = "RNA")

    # find label column
    cell_type_column <- "MERFISH.cell.type.annotation"

    if (!cell_type_column %in% colnames(RNAseurat_norm@meta.data)) {
      possible_columns <- c(
        "cell_type", "celltype", "Cell_Type", "annotation",
        "MERFISH_cell_type_annotation", "cluster", "seurat_clusters"
      )

      for (col in possible_columns) {
        if (col %in% colnames(RNAseurat_norm@meta.data)) {
          cell_type_column <- col
          break
        }
      }
    }

    if (cell_type_column %in% colnames(RNAseurat_norm@meta.data)) {
      cell_types <- RNAseurat_norm@meta.data[[cell_type_column]]
    } else {
      stop("Cell type annotation column not found.")
    }

    # Run SingleR
    pred_clust <- SingleR(
      test = merfish_data,
      ref = ref_data,
      labels = cell_types,
      de.method = "wilcox"
    )

    # Add predictions
    STseurat_norm$SingleR_labels <- pred_clust$labels

    # if you want one score column, use max score per cell
    STseurat_norm$SingleR_max_score <- apply(pred_clust$scores, 1, max)

    # save prediction
    series_dir <- file.path(base_out, paste0("series_", k))
    dir.create(series_dir, showWarnings = FALSE, recursive = TRUE)

    out_csv_A <- file.path(series_dir, "singler_predictions.csv")
    write.csv(as.data.frame(pred_clust), file = out_csv_A, row.names = TRUE)
  })

  end_time <- Sys.time()
  elapsed_sec <- as.numeric(difftime(end_time, start_time, units = "secs"))

  # peakRAM result usually contains Peak_RAM_Used_MiB
  peak_mem_mb <- ram_res$Peak_RAM_Used_MiB[1]

  perf_df <- rbind(
    perf_df,
    data.frame(
      series = k,
      time_sec = elapsed_sec,
      peak_mem_MB = peak_mem_mb
    )
  )

  cat("Finished series", k,
      "| time =", round(elapsed_sec, 2), "sec",
      "| peak memory =", round(peak_mem_mb, 2), "MB\n")
}
out_csv <- file.path(base_out, "runtimeSec_memoryMiB.csv")
write.csv(perf_df, out_csv, row.names = FALSE)


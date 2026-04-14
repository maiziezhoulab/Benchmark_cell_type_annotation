############################
## Libraries
############################
library(STdeconvolve)
library(Seurat)
library(patchwork)
library(zellkonverter)
library(stringr)
library(SingleCellExperiment)
library(peakRAM)
############################
## Load data
############################
sce <- readH5AD("/maiziezhou_lab2/yuling/Datasets/Development/5DPIs.h5ad")
############################
## Sanitize names (Seurat-safe)
############################
colnames(sce) <- make.unique(as.character(colnames(sce)))
rownames(sce) <- make.unique(as.character(rownames(sce)))

rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- rownames(sce)
gene_sanitized <- gsub("\\|", "-", gene_sanitized)
gene_sanitized <- gsub("_",  "-", gene_sanitized)
gene_sanitized <- make.unique(gene_sanitized)
rownames(sce) <- gene_sanitized

############################
## Ensure assays aligned
############################
stopifnot(all(c("counts", "X") %in% assayNames(sce)))
common <- intersect(
  rownames(assay(sce, "counts")),
  rownames(assay(sce, "X"))
)
sce <- sce[common, ]

############################
## Convert to Seurat
############################
seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]

############################
## Add spatial coordinates
############################
rd_name <- "spatial"
coords <- as.data.frame(reducedDim(sce, rd_name))
colnames(coords)[1:2] <- c("center_x", "center_y")
coords <- coords[colnames(seurat_obj), c("center_x", "center_y"), drop = FALSE]
seurat_obj <- AddMetaData(seurat_obj, metadata = coords)
############################
## Subset one section
############################
section_id <- "Injury_5DPI_rep3_SS200000147BL_D3"

per_section <- subset(
  seurat_obj,
  subset = Batch == section_id
)

pos   <- per_section@meta.data[, c("center_x", "center_y")]
cd    <- per_section@assays$originalexp$counts
annot <- per_section$Annotation

############################
## Output directory
############################
out_dir <- "/maiziezhou_lab2/yuling/label_Transfer/ST_deconvolve/regeneration"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

############################
## ---- STdeconvolve wrapped by peakRAM ----
############################
ram_info <- peakRAM({
  counts <- cleanCounts(
    counts = cd,
    min.lib.size = 10,
    min.reads = 0,
    min.detected = 0,
    verbose = TRUE
  )

  corpus <- restrictCorpus(
    counts,
    removeAbove = 1.0,
    removeBelow = 0.05,
    nTopOD = 500,
    plot = FALSE,
    verbose = TRUE
  )

  pixel_sums <- rowSums(corpus)
  corpus <- corpus[pixel_sums > 0, ]

  gene_sums <- colSums(corpus)
  corpus <- corpus[, gene_sums > 0]

  cat(
    "After filtering, retained",
    nrow(corpus), "pixels and",
    ncol(corpus), "genes\n"
  )

  ldas <- fitLDA(
    t(as.matrix(corpus)),
    Ks = seq(2, 9, by = 1),
    perc.rare.thresh = 0,
    plot = FALSE,
    verbose = TRUE
  )

  optLDA <- optimalModel(models = ldas, opt = "min")
  results <- getBetaTheta(
    optLDA,
    perc.filt = 0.05,
    betaScale = 1000
  )
  deconProp <<- results$theta
  deconGexp <<- results$beta
})

############################
## Extract runtime & memory
############################
runtime_sec <- ram_info$Elapsed_Time_sec
peak_memory_MiB <- ram_info$Peak_RAM_Used_MiB

############################
## Save deconvolution results
############################
write.csv(
  as.data.frame(deconProp),
  file.path(out_dir, paste0("deconProp_", section_id, ".csv")),
  row.names = TRUE,
  quote = FALSE
)

############################
## Save runtime & memory
############################
perf_df <- data.frame(
  runtime_sec = runtime_sec,
  peak_memory_MiB = peak_memory_MiB,
  stringsAsFactors = FALSE
)

write.csv(
  perf_df,
  file.path(out_dir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE,
  quote = FALSE
)

############################
## Done
############################
cat("=== STdeconvolve profiling finished ===\n")
cat("Runtime (sec):", runtime_sec, "\n")
cat("Peak memory (MiB):", round(peak_memory_MiB, 2), "\n")

############################
## Libraries (conda activate STdeconvolve)
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
sce <- readH5AD(
  "/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad"
)
############################
## Convert to Seurat
############################
seurat_obj <- as.Seurat(sce, counts = "raw", data = "X")

############################
## Add spatial coordinates
############################
rd_name <- "spatial"
coords <- as.data.frame(reducedDim(sce, rd_name))
colnames(coords)[1:2] <- c("center_x", "center_y")
coords <- coords[colnames(seurat_obj), c("center_x", "center_y"), drop = FALSE]
seurat_obj <- AddMetaData(seurat_obj, metadata = coords)

############################
## Subset query section
############################
section_id <- 19

per_section <- subset(
  seurat_obj,
  subset = n_section == section_id
)

pos   <- per_section@meta.data[, c("center_x", "center_y")]
cd    <- per_section@assays$originalexp$counts
annot <- per_section$annotation

############################
## Output directory
############################
out_dir <- "/maiziezhou_lab2/yuling/label_Transfer/ST_deconvolve/HumanLymph"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

############################
## ---- STdeconvolve wrapped by peakRAM ----
############################
ram_info <- peakRAM({

  counts <- cleanCounts(
    counts = cd,
    min.lib.size = 0,
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

  ## remove pixels with zero counts
  pixel_sums <- rowSums(corpus)
  corpus <- corpus[pixel_sums > 0, ]

  ## remove genes with zero counts
  gene_sums <- colSums(corpus)
  corpus <- corpus[, gene_sums > 0]

  cat(
    "After filtering, retained",
    nrow(corpus), "pixels and",
    ncol(corpus), "genes with non-zero counts\n"
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

  ## export to parent env
  deconProp <<- results$theta
  deconGexp <<- results$beta
})

############################
## Extract runtime & peak memory
############################
runtime_sec <- ram_info$Elapsed_Time_sec
peak_memory_MiB <- ram_info$Peak_RAM_Used_MiB

############################
## Save deconvolution result
############################
write.csv(
  as.data.frame(deconProp),
  file.path(out_dir, "deconProp_HumanLymph_section19.csv"),
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
cat("=== STdeconvolve (Human Lymph Node) finished ===\n")
cat("Runtime (sec):", runtime_sec, "\n")
cat("Peak memory (MiB):", round(peak_memory_MiB, 2), "\n")

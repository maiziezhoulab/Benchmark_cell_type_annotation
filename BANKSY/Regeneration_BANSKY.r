
############################################
## Libraries
############################################
library(Banksy)
library(SummarizedExperiment)
library(SpatialExperiment)
library(scuttle)
library(scater)
library(zellkonverter)
library(Seurat)
library(peakRAM)

############################################
## Paths
############################################
h5ad_path <- "/maiziezhou_lab2/yuling/datasets/Develop/5DPIs.h5ad"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/banksy/regeneration_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

############################################
## Load data (NOT benchmarked)
############################################
sce <- readH5AD(h5ad_path)

## Make names unique
colnames(sce) <- make.unique(as.character(colnames(sce)))
rownames(sce) <- make.unique(as.character(rownames(sce)))

## Sanitize gene symbols
rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- rownames(sce)
gene_sanitized <- gsub("\\|", "-", gene_sanitized)
gene_sanitized <- gsub("_", "-", gene_sanitized)
gene_sanitized <- make.unique(gene_sanitized)
rownames(sce) <- gene_sanitized

## Align assays
stopifnot(all(c("counts", "X") %in% assayNames(sce)))
common <- intersect(
  rownames(assay(sce, "counts")),
  rownames(assay(sce, "X"))
)
sce <- sce[common, ]

############################################
## Convert to Seurat (NOT benchmarked)
############################################
seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]

############################################
## Subset spatial section (NOT benchmarked)
############################################
per_section <- subset(
  seurat_obj,
  subset = Batch == "Injury_5DPI_rep3_SS200000147BL_D3"
)

dense_counts <- as.matrix(per_section@assays$originalexp$counts)

locations <- data.frame(
  sdimx = per_section@reductions$spatial@cell.embeddings[, 1],
  sdimy = per_section@reductions$spatial@cell.embeddings[, 2]
)

############################################
## Create SpatialExperiment (NOT benchmarked)
############################################
se <- SpatialExperiment(
  assay = list(counts = dense_counts),
  spatialCoords = as.matrix(locations)
)

se <- computeLibraryFactors(se)
assay(se, "normcounts") <- normalizeCounts(se, log = FALSE)

############################################
## BANKSY parameters
############################################
lambda <- c(0, 0.2)
k_geom <- c(15, 30)
n_clusters <- length(unique(per_section$Annotation))

############################################
## Benchmark BANKSY
############################################
res <- peakRAM({

  se_banksy <- Banksy::computeBanksy(
    se,
    assay_name = "normcounts",
    compute_agf = TRUE,
    k_geom = k_geom
  )

  se_banksy <- Banksy::runBanksyPCA(
    se_banksy,
    use_agf = TRUE,
    lambda = lambda
  )

  se_banksy <- Banksy::runBanksyUMAP(
    se_banksy,
    use_agf = TRUE,
    lambda = lambda
  )

  se_banksy <- Banksy::clusterBanksy(
    se_banksy,
    use_agf = TRUE,
    lambda = lambda,
    algo = "mclust",
    mclust.G = n_clusters
  )

  se_banksy <- Banksy::connectClusters(se_banksy)

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
## Save BANKSY output
############################################
cnames <- colnames(colData(se_banksy))
cnames <- cnames[grep("^clust", cnames)]
colData(se_banksy) <- cbind(
  colData(se_banksy),
  spatialCoords(se_banksy)
)

df <- as.data.frame(colData(se_banksy))

write.csv(
  df,
  file.path(outdir, "regeneration.csv"),
  row.names = TRUE
)

############################################
## Print summary
############################################
cat("BANKSY benchmark finished\n")
cat("Elapsed time (sec): ", runtime_sec, "\n")
cat("Peak RAM used (MiB): ", peak_mem_MiB, "\n")

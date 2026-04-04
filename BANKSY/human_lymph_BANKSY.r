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
## Load data
############################################
sce <- readH5AD(
  "/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad"
)

seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
per_section <- subset(
  seurat_obj,
  subset = n_section == 19)
dense_counts <- as.matrix(per_section@assays$originalexp$counts)

locations <- data.frame(
  sdimx = per_section@reductions$spatial@cell.embeddings[, 1],
  sdimy = per_section@reductions$spatial@cell.embeddings[, 2]
)

############################################
## Create SpatialExperiment
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
n_clusters <- length(unique(per_section$annotation))

############################################
## Output directory
############################################
out_dir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/banksy/HumanLymph"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

############################################
## Run BANKSY + peakRAM
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
## Save runtime & memory
############################################
summary_df <- data.frame(
  Elapsed_Time_sec = runtime_sec,
  Peak_RAM_Used_MiB = peak_mem_MiB
)

write.csv(
  summary_df,
  file.path(out_dir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

############################################
## Save BANKSY clustering output
############################################
colData(se_banksy) <- cbind(
  colData(se_banksy),
  spatialCoords(se_banksy)
)

df <- as.data.frame(colData(se_banksy))

write.csv(
  df,
  file.path(out_dir, "human_section19_BANKSY.csv"),
  row.names = TRUE
)

############################################
## Print summary
############################################
cat("BANKSY finished\n")
cat("Elapsed time (sec):", runtime_sec, "\n")
cat("Peak RAM used (MiB):", peak_mem_MiB, "\n")

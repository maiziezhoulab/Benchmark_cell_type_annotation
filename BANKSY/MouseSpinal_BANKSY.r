
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
# conda activate my-rdkit-env
############################################
## Paths
############################################
h5ad_path <- "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/banksy/output"
sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]
unique_Section <- unique(seurat_obj@meta.data$Section.ID)
selected_0503 <- grep("^0503", unique_Section, value = TRUE)
# select sections for analysis
selected_0503_clean <- setdiff(selected_0503, "0503_nan_nan")


############################################
## BANKSY parameters
############################################
lambda <- c(0, 0.2)
k_geom <- c(15, 30)

############################################
## Prepare output container
############################################
all_benchmark <- list()

############################################
## Loop over sections
############################################
for (sec_id in selected_0503_clean) {

  cat("\n==============================\n")
  cat("Running BANKSY for section:", sec_id, "\n")
  cat("==============================\n")

  ############################################
  ## Subset section
  ############################################
  per_section <- subset(
    seurat_obj,
    subset = Section.ID == sec_id
  )

  dense_counts <- as.matrix(per_section@assays$originalexp$counts)

  locations <- data.frame(
    sdimx = per_section@meta.data$center_x,
    sdimy = per_section@meta.data$center_y
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
  n_clusters <- length(
    unique(per_section$MERFISH.cell.type.annotation)
  )

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
  ## Save per-section benchmark
  ############################################
  sec_outdir <- file.path(outdir, sec_id)
  dir.create(sec_outdir, showWarnings = FALSE, recursive = TRUE)

  summary_df <- data.frame(
    Section_ID = sec_id,
    Elapsed_Time_sec = runtime_sec,
    Peak_RAM_Used_MiB = peak_mem_MiB
  )

  write.csv(
    summary_df,
    file.path(sec_outdir, "runtimeSec_memoryMiB.csv"),
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
    file.path(sec_outdir, paste0(sec_id, "_BANKSY.csv")),
    row.names = TRUE
  )

  ############################################
  ## Collect results
  ############################################
  all_benchmark[[sec_id]] <- summary_df

  cat("Finished section:", sec_id, "\n")
  cat("Elapsed time (sec):", runtime_sec, "\n")
  cat("Peak RAM (MiB):", peak_mem_MiB, "\n")
}

############################################
## Save combined benchmark table
############################################
all_benchmark_df <- do.call(rbind, all_benchmark)

write.csv(
  all_benchmark_df,
  file.path(outdir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

cat("\n=== ALL BANKSY BENCHMARKS FINISHED ===\n")


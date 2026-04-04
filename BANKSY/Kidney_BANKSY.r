

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
#h5ad_path <- "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/banksy/Kidney"

############################################
## BANKSY parameters
############################################
lambda <- c(0, 0.2)
k_geom <- c(15, 30)

############################################
## Prepare output container
############################################
all_benchmark <- list()
time_point <- c('Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6')
for (t in time_point){
############################################
## Loop over sections
############################################
    sections <- c('L', 'R')
    for (i in sections){
    # query data 
        sce <- readH5AD(paste0("/maiziezhou_lab2/yuling/datasets/Kidney/Xenium/time_",t, i,".h5ad"))
        per_section_spatial <- as.Seurat(sce, counts = "counts", data = "X")
        dense_counts <- as.matrix(per_section_spatial@assays$originalexp$counts)
        locations <- data.frame(
            sdimx = per_section_spatial@meta.data$x_centroid,
            sdimy = per_section_spatial@meta.data$y_centroid)

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
            unique(per_section_spatial$celltype_plot)
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
  sec_id <- paste0(t, i)
  summary_df <- data.frame(
    Section_ID = paste0(t, i),
    Elapsed_Time_sec = runtime_sec,
    Peak_RAM_Used_MiB = peak_mem_MiB
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
    file.path(outdir, paste0(t,i, "_BANKSY.csv")),
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

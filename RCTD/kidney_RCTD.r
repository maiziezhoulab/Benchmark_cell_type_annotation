library(spacexr)
library(Matrix)
library(future)
library(pryr)
library(dplyr)

## ---------------- Benchmark environment ----------------
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)
plan(sequential)
options(future.globals.maxSize = 5 * 1024^3)

## ---------------- Paths ----------------
refdir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_input"
reference_path <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_all_ref/SCRef.rds"
savedir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_output"
dir.create(savedir, showWarnings = FALSE, recursive = TRUE)

## ---------------- Parameters ----------------
CELL_MIN_INSTANCE <- 1
UMI_min <- 1
max_cores <- 1

query <- c("L", "R")
time_point <- c("Sham", "Hour4", "Hour12", "Day2", "Day14", "Week6")

## ---------------- Load reference once ----------------
reference_k <- readRDS(reference_path)

## ---------------- Benchmark result container ----------------
perf_list <- list()

## ======================= Main loop =======================
for (t in time_point) {
  for (q in query) {

    message("Running RCTD: ", t, " ", q)

    ## ---- Load spatial counts ----
    mSC <- readRDS(file.path(refdir_SingleR, paste0(t, q, "SCRaw.rds")))
    umi_per_spot <- Matrix::colSums(mSC@counts)

    ## ---- Create RCTD object (NOT timed) ----
    myRCTD_k <- create.RCTD(
      spatialRNA = mSC,
      reference = reference_k,
      max_cores = max_cores,
      UMI_min = UMI_min,
      CELL_MIN_INSTANCE = CELL_MIN_INSTANCE
    )

    myRCTD_k@config$UMI_min_sigma <- 55
    myRCTD_k@config$UMI_min <- min(myRCTD_k@config$UMI_min, 0)
    myRCTD_k@config$UMI_max <- Inf

    ## ---- Benchmark: run.RCTD ONLY ----
    gc()

    t_start <- Sys.time()

    mem_change <- pryr::mem_change({
      myRCTD_k <- run.RCTD(myRCTD_k, doublet_mode = "doublet")
    })

    t_end <- Sys.time()

    time_sec <- as.numeric(difftime(t_end, t_start, units = "secs"))
    mem_mib  <- as.numeric(mem_change) / 1024^2

    ## ---- Save outputs (NOT timed) ----
    savedir_Xenium <- file.path(savedir, paste0(t, q))
    dir.create(savedir_Xenium, showWarnings = FALSE, recursive = TRUE)

    saveRDS(myRCTD_k, file.path(savedir_Xenium, "mSC_RCTD.rds"))
    write.csv(
      myRCTD_k@results[[1]],
      file.path(savedir_Xenium, "RCTD_Xenium.csv"),
      row.names = TRUE
    )

    if (!is.null(myRCTD_k@results$weights)) {
      dense_weights <- as.matrix(myRCTD_k@results$weights)
      write.csv(
        dense_weights,
        file.path(savedir_Xenium, "RCTD_weights_all.csv")
      )
    }

    ## ---- Collect benchmark result ----
    perf_list[[length(perf_list) + 1]] <- data.frame(
      method = "RCTD",
      dataset = "Xenium",
      time_point = t,
      section = q,
      n_spatial_cells = ncol(mSC@counts),
      Time_sec = time_sec,
      Memory_MiB = mem_mib
    )
  }
}

## ======================= Save benchmark table =======================
perf_all <- bind_rows(perf_list)
write.csv(
  perf_all,
  file.path(savedir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE,
  quote = FALSE
)

message("Benchmark finished. Results saved to runtimeSec_memoryMiB.csv")

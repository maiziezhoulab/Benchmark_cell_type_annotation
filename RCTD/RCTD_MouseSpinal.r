# =======================
# RCTD runtime + peakRAM profiling (single run: k = 17)
# =======================
# conda activate my-rdkit-env

library(spacexr)
library(Matrix)
library(future)
library(peakRAM)   # <-- peak memory
library(tictoc)    # <-- timing

# -----------------------
# Thread control (fair benchmark)
# -----------------------
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1"
)

# -----------------------
# Paths
# -----------------------
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/ref_0503_F4_C"
datadir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"
savedir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/0503_F4_C_output"
dir.create(savedir, showWarnings = FALSE, recursive = TRUE)

# -----------------------
# Load spatial data (once)
# -----------------------
mSC <- readRDS(file.path(datadir_SingleR, "SCRaw.rds"))

# -----------------------
# Parameters
# -----------------------
k <- 17
CELL_MIN_INSTANCE <- 2
UMI_min <- 1
max_cores <- 4

# -----------------------
# Helper: build SC reference
# -----------------------
make_reference_from_series <- function(series_dir, min_UMI = 20, cell_min = 2) {

  counts_df <- read.csv(
    file.path(series_dir, "counts.csv"),
    header = TRUE, row.names = 1,
    colClasses = "character",
    stringsAsFactors = FALSE
  )

  counts_mat <- t(as.matrix(counts_df))
  storage.mode(counts_mat) <- "integer"

  meta_data <- read.table(
    file.path(series_dir, "meta_data.csv"),
    header = TRUE, row.names = 1, sep = ",",
    colClasses = "character",
    stringsAsFactors = FALSE
  )

  common_cells <- intersect(colnames(counts_mat), rownames(meta_data))
  counts_mat <- counts_mat[, common_cells, drop = FALSE]
  meta_data  <- meta_data[common_cells, , drop = FALSE]

  cell_types <- as.factor(meta_data$MERFISH.cell.type.annotation)
  levels(cell_types) <- gsub("/", "_", levels(cell_types))
  names(cell_types) <- rownames(meta_data)

  reference <- Reference(counts_mat, cell_types, min_UMI = min_UMI)

  ct_tab <- table(reference@cell_types)
  keep_types <- names(ct_tab[ct_tab >= cell_min])
  keep_cells <- names(reference@cell_types)[reference@cell_types %in% keep_types]

  reference@counts     <- reference@counts[, keep_cells, drop = FALSE]
  reference@cell_types <- droplevels(factor(reference@cell_types[keep_cells]))
  reference@nUMI       <- reference@nUMI[keep_cells]

  return(reference)
}

# =======================
# Run profiling (k = 17)
# =======================
series_dir <- file.path(series_base, paste0("series_", k))
stopifnot(dir.exists(series_dir))

message(">>> Running RCTD for series_", k)

# -----------------------
# Build reference
# -----------------------
reference_k <- make_reference_from_series(
  series_dir,
  min_UMI = 20,
  cell_min = CELL_MIN_INSTANCE
)

# -----------------------
# future setup (IMPORTANT for peakRAM correctness)
# -----------------------
plan(sequential)
options(future.globals.maxSize = 5 * 1024^3)

# -----------------------
# START PROFILING
# -----------------------
gc()

peak_res <- peakRAM({

  myRCTD_k <- create.RCTD(
    spatialRNA = mSC,
    reference = reference_k,
    max_cores = max_cores,
    UMI_min = UMI_min,
    CELL_MIN_INSTANCE = CELL_MIN_INSTANCE
  )

  myRCTD_k <- run.RCTD(
    myRCTD_k,
    doublet_mode = "doublet"
  )

})

runtime_sec  <- peak_res$Elapsed_Time_sec
# -----------------------
# Extract peak RAM (MiB)
# -----------------------
peak_mem_MiB <- peak_res$Peak_RAM_Used_MiB

# -----------------------
# Save outputs
# -----------------------
out_dir_k <- file.path(savedir, paste0("series_", k))
dir.create(out_dir_k, showWarnings = FALSE, recursive = TRUE)

saveRDS(
  myRCTD_k,
  file.path(out_dir_k, "mSC_RCTD.rds")
)

write.csv(
  myRCTD_k@results[[1]],
  file.path(out_dir_k, "RCTD_series17_results.csv"),
  row.names = FALSE
)

# -----------------------
# Save profiling summary
# -----------------------
profile_df <- data.frame(
  runtime_sec = runtime_sec,
  peak_memory_MiB = peak_mem_MiB
)

write.csv(
  profile_df,
  file.path(out_dir_k, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

message("=== DONE ===")
print(profile_df)

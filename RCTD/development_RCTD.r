library(spacexr)
library(Matrix)
library(future)
Sys.setenv("OMP_NUM_THREADS"="1", "OPENBLAS_NUM_THREADS"="1", "MKL_NUM_THREADS"="1")
# conda activate /home/huy21/anaconda3/envs/bindSC_R
## ----------------  ----------------
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_ref"
refdir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_input"
#datadir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"
savedir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Development_output"
dir.create(savedir, showWarnings = FALSE, recursive = TRUE)

## ---------------- raw  ----------------
mSC <- readRDS(file.path(refdir_SingleR,'SCRaw.rds'))
#  
CELL_MIN_INSTANCE <- 2
UMI_min <- 1
max_cores <- 4
## -------------   -------------
make_reference_from_series <- function(series_dir, min_UMI = 20, cell_min = CELL_MIN_INSTANCE) {
  # 
  counts_df <- read.csv(file.path(series_dir, "counts.csv"),
                        header = TRUE, row.names = 1,
                        colClasses = "character", stringsAsFactors = FALSE)
  counts_mat <- t(as.matrix(counts_df))
  storage.mode(counts_mat) <- "integer"
  #  
  meta_data <- read.table(file.path(series_dir, "meta_data.csv"),
                          header = TRUE, row.names = 1, sep = ",",
                          colClasses = "character", stringsAsFactors = FALSE)

  common_cells <- intersect(colnames(counts_mat), rownames(meta_data))
  if (length(common_cells) == 0) stop("No overlapping cells between counts and meta_data in: ", series_dir)
  counts_mat <- counts_mat[, common_cells, drop = FALSE]
  meta_data  <- meta_data[common_cells, , drop = FALSE]

  #  
  if (!"Annotation" %in% colnames(meta_data))
    stop("meta_data lack of original_clusters:", series_dir)
  cell_types <- meta_data$Annotation
  names(cell_types) <- rownames(meta_data)
  cell_types <- as.factor(cell_types)
  levels(cell_types) <- gsub("/", "_", levels(cell_types))  #  

  # create Reference
  reference <- Reference(counts_mat, cell_types, min_UMI = min_UMI)

  # filter cells by cell_min 
  ct_tab <- table(reference@cell_types)
  keep_types <- names(ct_tab[ct_tab >= cell_min])
  if (length(keep_types) == 0)
    stop("all of cell < ", cell_min, " :", series_dir)

  keep_cells <- names(reference@cell_types)[reference@cell_types %in% keep_types]
  reference@counts     <- reference@counts[, keep_cells, drop = FALSE]
  reference@cell_types <- droplevels(factor(reference@cell_types[keep_cells]))
  reference@nUMI       <- reference@nUMI[keep_cells]

  stopifnot(
    ncol(reference@counts) == length(reference@cell_types),
    length(reference@nUMI) == length(reference@cell_types)
  )
  return(reference)
}

## ---------------- build reference (outside benchmark) ----------------
reference_k <- make_reference_from_series(
  series_base,
  min_UMI = 20,
  cell_min = CELL_MIN_INSTANCE
)

saveRDS(reference_k, file.path(series_base, "SCRef.rds"))

## ---------------- benchmark RCTD ----------------
plan(sequential)
options(future.globals.maxSize = 5 * 1024^3)

gc()  # clean up before benchmark
mem_before <- sum(gc()[, "used"])
t_start <- Sys.time()

  myRCTD <- create.RCTD(
    mSC,
    reference_k,
    max_cores = max_cores,
    UMI_min = UMI_min,
    CELL_MIN_INSTANCE = CELL_MIN_INSTANCE
  )

  myRCTD@config$UMI_min_sigma <- 55
  myRCTD@config$UMI_min <- min(myRCTD@config$UMI_min, 20)
  myRCTD@config$UMI_max <- Inf

  myRCTD <- run.RCTD(myRCTD, doublet_mode = "doublet")
t_end <- Sys.time()
mem_after <- sum(gc()[, "used"])

runtime_sec <- as.numeric(difftime(t_end, t_start, units = "secs"))

# gc() unit = cells; 1 cell = 8 bytes
peak_mem_MiB <- max(mem_before, mem_after) * 8 / 1024^2

summary_df <- data.frame(
  Elapsed_Time_sec = runtime_sec,
  Peak_RAM_Used_MiB = peak_mem_MiB
)

write.csv(
  summary_df,
  file.path(savedir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)
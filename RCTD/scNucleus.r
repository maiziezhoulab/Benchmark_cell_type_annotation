library(spacexr)
library(Matrix)
library(future)
Sys.setenv("OMP_NUM_THREADS"="1", "OPENBLAS_NUM_THREADS"="1", "MKL_NUM_THREADS"="1")

## ----------------  ----------------
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_ref"
refdir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_input"
#datadir_SingleR <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"
savedir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_output"
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
  if (!"final_cluster_assignment" %in% colnames(meta_data))
    stop("meta_data lack of final_cluster_assignment:", series_dir)
  cell_types <- meta_data$final_cluster_assignment
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
# create SCRef
    reference_k <- make_reference_from_series(series_base, min_UMI = 20, cell_min = CELL_MIN_INSTANCE)
    saveRDS(reference_k, file.path(series_base, "SCRef.rds"))
    # use the following (80%) to set UMI_min_sigma 
    umi_per_spot <- Matrix::colSums(mSC@counts) 
    summary(umi_per_spot)
    quantile(umi_per_spot, c(0.5, 0.8, 0.9, 0.95))
     
    # run RCTD
    myRCTD_k <- create.RCTD(mSC, reference_k, max_cores = max_cores,
                            UMI_min = UMI_min, CELL_MIN_INSTANCE = CELL_MIN_INSTANCE)
    myRCTD_k@config$UMI_min_sigma <- 55
 
    myRCTD_k@config$UMI_min <- min(myRCTD_k@config$UMI_min, 20)    
    myRCTD_k@config$UMI_max <- Inf
    library(future)
    plan(sequential)
    options(future.globals.maxSize = 5 * 1024^3)
    myRCTD_k <- run.RCTD(myRCTD_k, doublet_mode = 'doublet')
    dir.create(savedir, showWarnings = FALSE, recursive = TRUE)
    saveRDS(myRCTD_k, file.path(savedir, "mSC_RCTD.rds"))
    write.csv(myRCTD_k@results[[1]], file.path(savedir,'RCTD_scNucleus.csv'), row.names = TRUE)
    #  
if (!is.null(myRCTD_k@results$weights)) {
  # 
  dense_weights <- as.matrix(myRCTD_k@results$weights)
  barcodes <- rownames(dense_weights)
  #  
  write.csv(dense_weights, file.path(savedir, "RCTD_weights_all.csv"))
}
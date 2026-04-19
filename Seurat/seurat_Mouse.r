library(Seurat)
library(Matrix)

# raw
datadir_raw <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"

# reference
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/ref_0503_F4_C"  #
series_range <- 1:17

#
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/0503_F4_C_outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# runtime / memory log (one row per series)
metrics_outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/0503_F4_C_output"
dir.create(metrics_outdir, showWarnings = FALSE, recursive = TRUE)
metrics_csv <- file.path(metrics_outdir, "runtimeSec_memoryMiB.csv")

## VmHWM in kB from /proc/self/status (Linux); NA if unavailable
read_vmhwm_kib <- function() {
  if (!file.exists("/proc/self/status")) {
    return(NA_real_)
  }
  st <- readLines("/proc/self/status", warn = FALSE)
  line <- grep("^VmHWM:", st, value = TRUE)
  if (!length(line)) {
    return(NA_real_)
  }
  parts <- strsplit(trimws(line[1]), "\\s+")[[1]]
  as.numeric(parts[2])
}

## Fallback: approximate RSS in MiB (Linux) or NA
read_vmrss_kib <- function() {
  if (!file.exists("/proc/self/status")) {
    return(NA_real_)
  }
  st <- readLines("/proc/self/status", warn = FALSE)
  line <- grep("^VmRSS:", st, value = TRUE)
  if (!length(line)) {
    return(NA_real_)
  }
  parts <- strsplit(trimws(line[1]), "\\s+")[[1]]
  as.numeric(parts[2])
}

## Fallback when /proc is missing: sum of gc() "max used" (Mb) — process-wide, not per-series
peak_mem_gc_mib <- function() {
  g <- gc(full = TRUE)
  if (!is.null(colnames(g)) && "max used" %in% colnames(g)) {
    return(sum(as.numeric(g[, "max used"]), na.rm = TRUE))
  }
  sum(as.numeric(g[, ncol(g)]), na.rm = TRUE)
}

## =========  create seurat object =========
make_seurat_from_counts_meta <- function(dir_path) {
  # counts：
  counts_df <- read.csv(file.path(dir_path, "counts.csv"),
                        header = TRUE, row.names = 1,
                        colClasses = "character", stringsAsFactors = FALSE)
  counts_mat <- t(counts_df)
  storage.mode(counts_mat) <- "integer"

  # meta：
  meta_data <- read.table(file.path(dir_path, "meta_data.csv"),
                          header = TRUE, row.names = 1, sep = ",",
                          colClasses = "character", stringsAsFactors = FALSE)
  #
  #common_cells <- intersect(colnames(counts_mat), rownames(meta_data))
  #if (length(common_cells) == 0) {
    #stop("No overlapping cells between counts and meta_data in: ", dir_path)
  #}
  #counts_mat <- counts_mat[, common_cells, drop = FALSE]
  #meta_data  <- meta_data[common_cells, , drop = FALSE]

  obj <- CreateSeuratObject(counts = counts_mat, meta.data = meta_data)
  return(obj)
}

## =========   =========
raw <- make_seurat_from_counts_meta(datadir_raw)

# process raw
raw <- NormalizeData(raw)
raw <- FindVariableFeatures(raw)
raw <- ScaleData(raw)
raw <- RunPCA(raw)

metrics_rows <- list()

## ========= 2) loop reference =========
for (k in series_range) {
  series_dir <- file.path(series_base, paste0("series_", k))
  if (!dir.exists(series_dir)) {
    message("directory not exist：", series_dir)
    next
  }
  message(">>> process reference: ", basename(series_dir))

  gc(full = TRUE)
  hwm0 <- read_vmhwm_kib()
  rss0 <- read_vmrss_kib()
  t0 <- proc.time()

  # read  reference
  ref <- make_seurat_from_counts_meta(series_dir)

  #
  if (!"MERFISH.cell.type.annotation" %in% colnames(ref@meta.data)) {
    stop("meta_data lack of MERFISH.cell.type.annotation:", series_dir)
  }

  # preprocess reference
  ref <- NormalizeData(ref)
  ref <- FindVariableFeatures(ref)
  ref <- ScaleData(ref)
  ref <- RunPCA(ref)

  # Find anchors
  anchors <- FindTransferAnchors(
    reference = ref,
    query = raw,
    dims = 1:30,
    reference.reduction = "pca"
  )

  # Transfer labels
  predictions <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data$MERFISH.cell.type.annotation,
    dims = 1:30
  )

  #
  outfile <- file.path(outdir, paste0("seurat_", basename(series_dir), "_pred.csv"))
  write.csv(predictions, file = outfile)

  elapsed_sec <- as.numeric(proc.time() - t0)[3]
  gc(full = TRUE)
  hwm1 <- read_vmhwm_kib()
  rss1 <- read_vmrss_kib()

  peak_mib <- NA_real_
  if (!is.na(hwm0) && !is.na(hwm1)) {
    peak_mib <- pmax(0, (hwm1 - hwm0) / 1024)
  } else if (!is.na(rss0) && !is.na(rss1)) {
    peak_mib <- pmax(0, (rss1 - rss0) / 1024)
  } else {
    peak_mib <- peak_mem_gc_mib()
  }

  metrics_rows[[length(metrics_rows) + 1]] <- data.frame(
    series = k,
    runtime_sec = elapsed_sec,
    peak_memory_MiB = peak_mib,
    stringsAsFactors = FALSE
  )

  rm(ref, anchors, predictions)
  gc(full = TRUE)

  message("<<< finish ", basename(series_dir), " result ", outfile,
          " | ", round(elapsed_sec, 3), " sec, ~", round(peak_mib, 2), " MiB (delta)")
}

if (length(metrics_rows)) {
  metrics_df <- do.call(rbind, metrics_rows)
  write.csv(metrics_df, file = metrics_csv, row.names = FALSE)
  message("Wrote metrics: ", metrics_csv)
} else {
  warning("No series processed; metrics CSV not written.")
}

message("all complete !")

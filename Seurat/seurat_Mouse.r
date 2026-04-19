library(Seurat)
library(Matrix)

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

## Fallback when /proc is missing
peak_mem_gc_mib <- function() {
  g <- gc(full = TRUE)
  if (!is.null(colnames(g)) && "max used" %in% colnames(g)) {
    return(sum(as.numeric(g[, "max used"]), na.rm = TRUE))
  }
  sum(as.numeric(g[, ncol(g)]), na.rm = TRUE)
}

get_script_path <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", ca, value = TRUE)
  if (length(f)) {
    return(normalizePath(sub("^--file=", "", f[1]), mustWork = FALSE))
  }
  NA_character_
}

## ========= paths =========
datadir_raw <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/Raw_Spatial_0503_F4_C"
series_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/ref_0503_F4_C"
series_range <- 1:17

outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/0503_F4_C_outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

metrics_outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/0503_F4_C_output"
dir.create(metrics_outdir, showWarnings = FALSE, recursive = TRUE)
metrics_csv <- file.path(metrics_outdir, "runtimeSec_memoryMiB.csv")

## =========  create seurat object =========
make_seurat_from_counts_meta <- function(dir_path) {
  counts_df <- read.csv(file.path(dir_path, "counts.csv"),
                        header = TRUE, row.names = 1,
                        colClasses = "character", stringsAsFactors = FALSE)
  counts_mat <- t(counts_df)
  storage.mode(counts_mat) <- "integer"

  meta_data <- read.table(file.path(dir_path, "meta_data.csv"),
                          header = TRUE, row.names = 1, sep = ",",
                          colClasses = "character", stringsAsFactors = FALSE)

  obj <- CreateSeuratObject(counts = counts_mat, meta.data = meta_data)
  return(obj)
}

## ========= one series (full pipeline in fresh process when called via Rscript --series=k) =========
run_one_series <- function(k) {
  series_dir <- file.path(series_base, paste0("series_", k))
  if (!dir.exists(series_dir)) {
    message("directory not exist：", series_dir)
    return(invisible(NULL))
  }

  t0 <- proc.time()

  raw <- make_seurat_from_counts_meta(datadir_raw)
  raw <- NormalizeData(raw)
  raw <- FindVariableFeatures(raw)
  raw <- ScaleData(raw)
  raw <- RunPCA(raw)

  message(">>> process reference: ", basename(series_dir))

  ref <- make_seurat_from_counts_meta(series_dir)

  if (!"MERFISH.cell.type.annotation" %in% colnames(ref@meta.data)) {
    stop("meta_data lack of MERFISH.cell.type.annotation: ", series_dir)
  }

  ref <- NormalizeData(ref)
  ref <- FindVariableFeatures(ref)
  ref <- ScaleData(ref)
  ref <- RunPCA(ref)

  anchors <- FindTransferAnchors(
    reference = ref,
    query = raw,
    dims = 1:30,
    reference.reduction = "pca"
  )

  predictions <- TransferData(
    anchorset = anchors,
    refdata = ref@meta.data$MERFISH.cell.type.annotation,
    dims = 1:30
  )

  outfile <- file.path(outdir, paste0("seurat_", basename(series_dir), "_pred.csv"))
  write.csv(predictions, file = outfile)

  elapsed_sec <- as.numeric(proc.time() - t0)[3]
  gc(full = TRUE)

  hwm_kib <- read_vmhwm_kib()
  peak_mib <- if (!is.na(hwm_kib)) {
    hwm_kib / 1024
  } else {
    peak_mem_gc_mib()
  }

  message("<<< finish ", basename(series_dir), " | ", round(elapsed_sec, 3), " sec | peak ~",
          round(peak_mib, 2), " MiB (process VmHWM)")

  cat(sprintf(
    "SEURAT_METRICS\t%d\t%.10f\t%.10f\n",
    as.integer(k), elapsed_sec, peak_mib
  ))

  invisible(list(
    series = as.integer(k),
    runtime_sec = elapsed_sec,
    peak_memory_MiB = peak_mib
  ))
}

## ========= CLI =========
args_trail <- commandArgs(trailingOnly = TRUE)
series_idx <- grep("^--series=", args_trail, value = TRUE)

if (length(series_idx)) {
  k <- as.integer(sub("^--series=", "", series_idx[1]))
  if (is.na(k)) {
    stop("Invalid --series= value")
  }
  run_one_series(k)
  quit(save = "no", status = 0)
}

## ========= batch: fresh R child per series → comparable VmHWM peak =========
rscript <- file.path(R.home("bin"), "Rscript")
script_path <- get_script_path()
if (is.na(script_path) || !nzchar(script_path) || !file.exists(script_path)) {
  stop("Run this script with: Rscript seurat_Mouse.r  (need --file= path for batch mode)")
}

metrics_rows <- list()

for (k in series_range) {
  series_dir <- file.path(series_base, paste0("series_", k))
  if (!dir.exists(series_dir)) {
    message("directory not exist：", series_dir)
    next
  }

  so <- system2(
    rscript,
    args = c(script_path, paste0("--series=", k)),
    stdout = TRUE,
    stderr = FALSE
  )
  st <- attr(so, "status")
  if (!is.null(st) && st != 0) {
    warning("Rscript exit ", st, " for series ", k)
    next
  }

  line <- grep("^SEURAT_METRICS\t", so, value = TRUE)
  if (!length(line)) {
    warning("No SEURAT_METRICS line for series ", k)
    next
  }

  parts <- strsplit(line[1], "\t", fixed = TRUE)[[1]]
  if (length(parts) < 4) {
    warning("Bad metrics line for series ", k)
    next
  }

  metrics_rows[[length(metrics_rows) + 1]] <- data.frame(
    series = as.integer(parts[2]),
    runtime_sec = as.numeric(parts[3]),
    peak_memory_MiB = as.numeric(parts[4]),
    stringsAsFactors = FALSE
  )
}

if (length(metrics_rows)) {
  metrics_df <- do.call(rbind, metrics_rows)
  write.csv(metrics_df, file = metrics_csv, row.names = FALSE)
  message("Wrote metrics: ", metrics_csv)
} else {
  warning("No series processed; metrics CSV not written.")
}

message("all complete !")

library(Seurat)
library(Matrix)

outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Seurat/scNucleus_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
metrics_csv <- file.path(outdir, "runtimeSec_memoryMiB.csv")

## Peak resident set (Linux): VmHWM in kB → MiB; NA if unavailable
read_vmhwm_mib <- function() {
  if (!file.exists("/proc/self/status")) {
    return(NA_real_)
  }
  st <- readLines("/proc/self/status", warn = FALSE)
  line <- grep("^VmHWM:", st, value = TRUE)
  if (!length(line)) {
    return(NA_real_)
  }
  parts <- strsplit(trimws(line[1]), "\\s+")[[1]]
  as.numeric(parts[2]) / 1024
}

peak_mem_gc_mib <- function() {
  g <- gc(full = TRUE)
  if (!is.null(colnames(g)) && "max used" %in% colnames(g)) {
    return(sum(as.numeric(g[, "max used"]), na.rm = TRUE))
  }
  sum(as.numeric(g[, ncol(g)]), na.rm = TRUE)
}

## Wall time: from first data read through writing predictions
t0 <- proc.time()

refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_ref"
counts <- read.csv(file.path(refdir, "counts.csv"),
                   header = TRUE,
                   row.names = 1,
                   colClasses = "character",
                   stringsAsFactors = FALSE)

counts_t <- t(counts)
storage.mode(counts_t) <- "integer"
counts <- counts_t

meta_data <- read.table(file.path(refdir, "meta_data.csv"),
                        header = TRUE,
                        row.names = 1,
                        sep = ",",
                        colClasses = "character",
                        stringsAsFactors = FALSE)

ref <- CreateSeuratObject(counts = counts, meta.data = meta_data)

############################################ Load spatial transcriptomics data
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/scNucleus_input"

counts <- read.csv(file.path(datadir, "counts.csv"),
                   header = TRUE,
                   row.names = 1,
                   colClasses = "character",
                   stringsAsFactors = FALSE)

counts_t <- t(counts)
storage.mode(counts_t) <- "integer"
counts <- counts_t

meta_data <- read.table(file.path(datadir, "meta_data.csv"),
                        header = TRUE,
                        row.names = 1,
                        sep = ",",
                        colClasses = "character",
                        stringsAsFactors = FALSE)

raw <- CreateSeuratObject(counts = counts, meta.data = meta_data)

############################################

# Preprocess reference
ref <- NormalizeData(ref)
ref <- FindVariableFeatures(ref)
ref <- ScaleData(ref)
ref <- RunPCA(ref)

# Preprocess query
raw <- NormalizeData(raw)
raw <- FindVariableFeatures(raw)
raw <- ScaleData(raw)
raw <- RunPCA(raw)

##############################

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
  refdata = ref@meta.data$final_cluster_assignment,
  dims = 1:30
)

pred_path <- file.path(outdir, "seurat_Nucleus.csv")
write.csv(predictions, file = pred_path)

elapsed_sec <- as.numeric(proc.time() - t0)[3]
gc(full = TRUE)
peak_mib <- read_vmhwm_mib()
if (is.na(peak_mib)) {
  peak_mib <- peak_mem_gc_mib()
}

write.csv(
  data.frame(Time = elapsed_sec, Memory = peak_mib),
  file = metrics_csv,
  row.names = FALSE
)

message("Saved predictions: ", pred_path)
message("Saved metrics (Time = wall sec, Memory ≈ peak MiB): ", metrics_csv)

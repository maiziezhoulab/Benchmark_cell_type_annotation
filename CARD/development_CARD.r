library(Seurat)
library(SeuratData)
library(CARD)
library(zellkonverter)
library(SingleCellExperiment)
library(dplyr)

# ===============================
# Load data (NOT benchmarked)
# ===============================
sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/Development.h5ad")

colnames(sce) <- make.unique(colnames(sce))
rownames(sce) <- make.unique(rownames(sce))

rowData(sce)$gene_original <- rownames(sce)
gene_sanitized <- gsub("[|_]", "-", rownames(sce))
rownames(sce) <- make.unique(gene_sanitized)

stopifnot(all(c("counts", "X") %in% assayNames(sce)))
common <- intersect(rownames(assay(sce, "counts")),
                    rownames(assay(sce, "X")))
sce <- sce[common, ]

seurat_obj <- as.Seurat(sce, counts = "counts", data = "X")
seurat_obj[["RNA"]] <- seurat_obj[["originalexp"]]
DefaultAssay(seurat_obj) <- "RNA"

# ===============================
# Spatial coordinates
# ===============================
coords <- as.data.frame(reducedDim(sce, "spatial"))
colnames(coords)[1:2] <- c("center_x", "center_y")
coords <- coords[colnames(seurat_obj), ]
seurat_obj <- AddMetaData(seurat_obj, coords)

# ===============================
# Reference & query (NOT benchmarked)
# ===============================
total_data <- subset(
  seurat_obj,
  subset = Batch == "Stage54_telencephalon_rep2_DP8400015649BRD6_2"
)

per_section_spatial <- subset(
  seurat_obj,
  subset = Batch == "Stage44_telencephalon_rep2_FP200000239BL_E4"
)

sc_count <- total_data@assays$originalexp$data
sc_meta <- total_data@meta.data |>
  dplyr::rename(cellType = Annotation)
sc_meta$sampleInfo <- "sample1"

spatial_count <- per_section_spatial@assays$originalexp$data
meta <- per_section_spatial[[]]
spatial_location <- meta[, c("center_x", "center_y")]
colnames(spatial_location) <- c("x", "y")

# ===============================
# BENCHMARK START
# ===============================

library(peakRAM)

res <- peakRAM({

  CARD_obj <- createCARDObject(
    sc_count = sc_count,
    sc_meta = sc_meta,
    spatial_count = spatial_count,
    spatial_location = spatial_location,
    ct.varname = "cellType",
    ct.select = unique(sc_meta$cellType),
    sample.varname = "sampleInfo",
    minCountGene = 0,
    minCountSpot = 0
  )

  CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)

})
runtime_sec <- res$Elapsed_Time_sec
peak_mem_MiB <- res$Peak_RAM_Used_MiB
# ===============================
# Save results
# ===============================
out_dir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/Development_output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write.csv(
  CARD_obj@Proportion_CARD,
  file.path(out_dir, "CARD_predictions_1.csv"),
  quote = FALSE
)

summary_df <- data.frame(
  Elapsed_Time_sec = runtime_sec,
  Peak_RAM_Used_MiB = peak_mem_MiB
)

write.csv(
  summary_df,
  file.path(out_dir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

print(summary_df)

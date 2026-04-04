############################
## Environment & Libraries
############################
library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(CARD)
library(zellkonverter)
library(SingleCellExperiment)
library(readxl)
library(stringr)
library(peakRAM)

gc()

############################
## Load data
############################
sce <- readH5AD(
  "/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad"
)

seurat_obj <- as.Seurat(sce, counts = "X", data = "X")

############################
## Add spatial coordinates
############################
rd_name <- "spatial"
coords <- as.data.frame(reducedDim(sce, rd_name))
colnames(coords)[1:2] <- c("center_x", "center_y")
coords <- coords[colnames(seurat_obj), c("center_x", "center_y"), drop = FALSE]

seurat_obj <- AddMetaData(seurat_obj, metadata = coords)

############################
## Reference & Query split
############################
# reference
total_data <- subset(
  seurat_obj,
  subset = n_section == 6
)

# query
per_section_spatial <- subset(
  seurat_obj,
  subset = n_section == 19
)

############################
## Spatial counts & location
############################
spatial_count <- per_section_spatial@assays$originalexp$data

meta <- per_section_spatial[[]]
centers <- meta[, c("center_x", "center_y"), drop = FALSE]

spatial_location <- cbind(
  cell_id = rownames(centers),
  setNames(centers, c("x", "y"))
)

rownames(spatial_location) <- NULL
spatial_location <- tibble::column_to_rownames(
  spatial_location,
  var = "cell_id"
)

############################
## Single-cell reference
############################
sc_count <- total_data@assays$originalexp$data

sc_meta <- total_data@meta.data
sc_meta <- dplyr::rename(sc_meta, cellType = annotation)
sc_meta$sampleInfo <- "sample1"

############################
## CARD + runtime + memory
############################
gc()  

peak_res <- peakRAM({

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

  CARD_obj <- CARD_deconvolution(
    CARD_object = CARD_obj
  )
})

############################
## Output paths
############################
outdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/HumanLymph_output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

############################
## Save results
############################
write.csv(
  CARD_obj@Proportion_CARD,
  file.path(outdir, "CARD_predictions.csv"),
  quote = FALSE
)

write.csv(
  peak_res,
  file.path(outdir, "CARD_runtime_memory.csv"),
  row.names = FALSE
)

############################
## Print summary
############################
print("===== CARD Benchmark Summary =====")
print(peak_res)

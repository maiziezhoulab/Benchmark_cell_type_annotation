library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
# conda env : conda activate r-gis
#library(SeuratWrappers)
library(CARD)
#library(viridis)
library(zellkonverter)
library(SingleCellExperiment)
library(readxl)
library(stringr)
sce <- readH5AD("/maiziezhou_lab2/yuling/datasets/obj_integrated_sc_nucleus.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
total_data <- seurat_obj
rd_name <-  "spatial"
#########################################
sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
per_section_spatial <- subset(seurat_obj, 
                        subset = Section.ID == '0503_F4_C')   
spatial_count <- per_section_spatial@assays$originalexp$data
#----------- spatial location data must be in the format of data frame 
meta <- per_section_spatial[[]]
centers <- meta[, c("center_x", "center_y"), drop = FALSE]
spatial_location <- cbind(cell_id = rownames(centers),
                 setNames(centers, c("x", "y")))
rownames(spatial_location) <- NULL
spatial_location <- tibble::column_to_rownames(spatial_location, var = "cell_id")
#----------------- single cell data 
sc_count <- total_data@assays$originalexp$data
sc_meta <- total_data@meta.data
sc_meta <- dplyr::rename(sc_meta, cellType = final_cluster_assignment)
sc_meta$sampleInfo = "sample1"
CARD_obj = createCARDObject(
	sc_count = sc_count,
	sc_meta = sc_meta,
	spatial_count = spatial_count,
	spatial_location = spatial_location,
	ct.varname = "cellType",
	ct.select = unique(sc_meta$cellType),
	sample.varname = "sampleInfo",
	minCountGene = 0,
	minCountSpot = 0)
CARD_obj = CARD_deconvolution(CARD_object = CARD_obj)
write.csv(CARD_obj@Proportion_CARD, "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/scNucleus_output/CARD_predictions.csv", quote = FALSE)
#------
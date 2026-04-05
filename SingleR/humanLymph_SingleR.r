library(reticulate)
library(Seurat)
library(anndata)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)


# -----------------------------
# conda activate /home/huy21/anaconda3/envs/bindSC_R
# -----------------------------
sce <- readH5AD("/maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 

per_section_ref <- subset(seurat_obj, subset = n_section == 6)
counts <- per_section_ref@assays$originalexp@counts
cell_metadata <- as.data.frame(per_section_ref@meta.data)

RNAseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
RNAseurat_norm <- NormalizeData(RNAseurat)
RNAseurat_norm <- ScaleData(RNAseurat_norm)

per_section <- subset(seurat_obj, subset = n_section == 19)
counts <- per_section@assays$originalexp@counts
cell_metadata <- as.data.frame(per_section@meta.data)

STseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
STseurat_norm <- NormalizeData(STseurat)
STseurat_norm <- ScaleData(STseurat_norm)

ref_data <- as.SingleCellExperiment(RNAseurat_norm)
merfish_data <- as.SingleCellExperiment(STseurat_norm)

cell_type_column <- "annotation"
cell_types <- RNAseurat_norm@meta.data[[cell_type_column]]

# -----------------------------
# SingleR：runtime + peak memory
# -----------------------------
# -----------------------------
# Runtime + memory (base R)
# -----------------------------
mem_log <- tempfile()
Rprofmem(mem_log)

start_time <- Sys.time()

pred_clust <- SingleR(
  test = merfish_data,
  ref = ref_data,
  labels = cell_types,
  de.method = "wilcox"
)

end_time <- Sys.time()
Rprofmem(NULL)

# runtime
runtime_sec <- as.numeric(difftime(end_time, start_time, units = "secs"))

# peak memory
mem_lines <- readLines(mem_log)
mem_bytes <- as.numeric(sub(":.*", "", mem_lines))
peak_mem_mib <- max(mem_bytes, na.rm = TRUE) / 1024^2

cat("Runtime (sec):", runtime_sec, "\n")
cat("Peak memory (MiB):", peak_mem_mib, "\n")
STseurat_norm$SingleR_labels <- pred_clust$labels
STseurat_norm$SingleR_scores <- pred_clust$scores

write.csv(
  pred_clust,
  file = "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/HumanLymph_output/Results.csv",
  row.names = TRUE
)
benchmark_df <- data.frame(
  Elapsed_Time_sec = runtime_sec,
  Peak_RAM_Used_MiB = peak_mem_mib
)

write.csv(
  benchmark_df,
  "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/HumanLymph_output/runtimeSec_memoryMiB.csv",
  row.names = FALSE
)

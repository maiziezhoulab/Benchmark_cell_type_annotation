############################
## 0. Packages
############################
library(STdeconvolve)
library(Seurat)
library(zellkonverter)
library(stringr)
library(peakRAM)

############################
## 1. Load data
############################
sce <- readH5AD(
  "/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad"
)
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 

############################
## 2. Select sections
############################
unique_Section <- unique(seurat_obj@meta.data$Section.ID)

selected_0503 <- grep("^0503", unique_Section, value = TRUE)
selected_0503_clean <- setdiff(selected_0503, "0503_nan_nan")

############################
## 3. Output directory
############################
out_dir <- "/maiziezhou_lab2/yuling/label_Transfer/ST_deconvolve/output"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

############################
## 4. Record table
############################
records <- list()

############################
## 5. Main loop
############################
for (i in selected_0503_clean) {

  message("====================================")
  message("Running section: ", i)
  message("====================================")

  res <- tryCatch({

    ## ---- runtime + peak memory ----
    timing <- system.time({
      peak <- peakRAM({

        ## ---- subset section ----
        per_section <- subset(
          seurat_obj,
          subset = Section.ID == i
        )

        if (ncol(per_section) == 0) {
          stop("No cells in this section")
        }

        ## ---- extract counts (integer) ----
        cd <- GetAssayData(per_section, slot = "counts")
        cd <- round(cd)

        ## ---- STdeconvolve preprocessing ----
        counts <- cleanCounts(
          counts = cd,
          min.lib.size = 10,
          min.reads = 0,
          min.detected = 0,
          verbose = FALSE
        )

        corpus <- restrictCorpus(
          counts,
          removeAbove = 1.0,
          removeBelow = 0.05,
          nTopOD = 500,
          plot = FALSE,
          verbose = FALSE
        )

        ## ---- remove empty pixels / genes ----
        corpus <- corpus[rowSums(corpus) > 0, ]
        corpus <- corpus[, colSums(corpus) > 0]

        message(
          "After filtering: ",
          nrow(corpus), " pixels × ",
          ncol(corpus), " genes"
        )

        ## ---- LDA ----
        ldas <- fitLDA(
          t(as.matrix(corpus)),
          Ks = 2:9,
          perc.rare.thresh = 0,
          plot = FALSE,
          verbose = FALSE
        )

        optLDA <- optimalModel(ldas, opt = "min")

        ## ---- extract results ----
        results <- getBetaTheta(
          optLDA,
          perc.filt = 0.05,
          betaScale = 1000
        )

        deconProp <- results$theta

        ## ---- save deconvolution result ----
        write.csv(
          as.data.frame(deconProp),
          file.path(out_dir, sprintf("deconProp_%s.csv", i)),
          row.names = TRUE,
          quote = FALSE
        )
      })
    })

    ## ---- record metrics ----
    data.frame(
      method = "STdeconvolve",
      dataset = "MERFISH_spinal",
      section = i,
      n_pixels = ncol(per_section),
      n_genes = nrow(cd),
      runtime_sec = as.numeric(timing["elapsed"]),
      peak_memory_MiB = peak$Peak_RAM_Used_MiB,
      status = "success"
    )

  }, error = function(e) {
    message("Reason: ", e$message)

    data.frame(
      method = "STdeconvolve",
      dataset = "MERFISH_spinal",
      section = i,
      n_pixels = NA,
      n_genes = NA,
      runtime_sec = NA,
      peak_memory_MiB = NA,
      status = "failed"
    )
  })

  records[[length(records) + 1]] <- res
}

############################
## 6. Save runtime + memory summary
############################
runtime_df <- do.call(rbind, records)

write.csv(
  runtime_df,
  file.path(out_dir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

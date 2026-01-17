library(STdeconvolve)
library(Seurat)
library(zellkonverter)
library(stringr)
library(peakRAM)

out_dir <- "/maiziezhou_lab2/yuling/label_Transfer/ST_deconvolve/Kidney"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

time_point <- c("Sham", "Hour4", "Hour12", "Day2", "Day14", "Week6")
sections <- c("L", "R")

records <- list()

for (t in time_point) {
  for (i in sections) {
    sec_id <- paste0(t, i)
      sce <- readH5AD(paste0('/maiziezhou_lab2/yuling/label_Transfer/kidney_data/time_', sec_id,'.h5ad'))
      per_section <- as.Seurat(sce, counts = "counts", data = "counts")
        cd <- per_section@assays$originalexp$counts

        counts <- cleanCounts(
          counts = cd,
          min.lib.size = 0,
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

        corpus <- corpus[rowSums(corpus) > 0, ]
        corpus <- corpus[, colSums(corpus) > 0]
    timing <- system.time({
      peak <- peakRAM({

        ldas <- fitLDA(
          t(as.matrix(corpus)),
          Ks = seq(2, 9, 1),
          perc.rare.thresh = 0,
          plot = FALSE,
          verbose = FALSE
        )

        optLDA <- optimalModel(ldas, opt = "min")

        results <- getBetaTheta(
          optLDA,
          perc.filt = 0.05,
          betaScale = 1000
        )
      })
    })

    # ---------------- record metrics ----------------
    records[[length(records) + 1]] <- data.frame(
      method        = "STdeconvolve",
      dataset       = "Kidney_Xenium",
      time_point    = t,
      section       = i,
      runtime_sec   = as.numeric(timing["elapsed"]),
      peak_memory_MiB = peak$Peak_RAM_Used_MiB
    )

    # optional: save output
    write.csv(
      as.data.frame(results$theta),
      file.path(out_dir, sprintf("deconProp_%s_%s.csv", t, i)),
      row.names = TRUE,
      quote = FALSE
    )
  }
}

# ---------------- save summary ----------------
runtime_df <- do.call(rbind, records)
write.csv(
  runtime_df,
  file.path(out_dir, "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)

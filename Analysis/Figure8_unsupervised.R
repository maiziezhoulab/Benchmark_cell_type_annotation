library(ggplot2)
library(ggforce)
library(RColorBrewer)
library(dplyr)
library(scales)
library(cowplot)
library(tibble)
library(tidyr)

#----------------------------
add_column_if_missing <- function(df, ...) {
  column_values <- list(...)
  for (nm in names(column_values)) {
    default_val <- rep(column_values[[nm]], nrow(df))
    if (nm %in% colnames(df)) {
      df[[nm]] <- ifelse(is.na(df[[nm]]), default_val, df[[nm]])
    } else {
      df[[nm]] <- default_val
    }
  }
  df
}
##################################################################
get_brewer <- function(name, n = 9, reverse = TRUE) {
  if (requireNamespace("RColorBrewer", quietly = TRUE)) {
    info <- RColorBrewer::brewer.pal.info
    if (name %in% rownames(info)) {
      nmax <- info[name, "maxcolors"]
      pal  <- RColorBrewer::brewer.pal(min(n, nmax), name)
      return(if (reverse) rev(pal) else pal)
    }
  }
  scales::hue_pal()(n)
}

# ---------------------------------------------
# Plot function (now supports a "method" column)
# ---------------------------------------------
plot_scmmib_summary <- function(
    data, row_info, column_info, palettes,
    rank = FALSE, rect_normalize = TRUE, extend_figure = FALSE
) {
  row_height <- 1.1
  row_space  <- .2
  col_space  <- .2
  col_bigspace <- 0.6
  segment_data <- tibble()
  
  # Rows
  if (!"group" %in% colnames(row_info) || all(is.na(row_info$group))) row_info$group <- ""
  row_pos <- row_info %>%
    dplyr::group_by(group) %>%
    dplyr::mutate(group_i = dplyr::row_number()) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      row_i = dplyr::row_number(),
      colour_background = group_i %% 2 == 1,
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      ysep = ifelse(do_spacing, row_height + 2 * row_space, row_space),
      y = - (row_i * row_height + cumsum(ysep)),
      ymin = y - row_height / 2,
      ymax = y + row_height / 2
    )
  
  # Columns
  if (!"group" %in% colnames(column_info) || all(is.na(column_info$group))) column_info$group <- ""
  column_info <- column_info %>% add_column_if_missing(width = 1.1, overlay = FALSE)
  
  column_pos <- column_info %>%
    dplyr::mutate(
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      xsep = dplyr::case_when(
        overlay ~ c(0, -head(width, -1)),
        do_spacing ~ col_bigspace,
        TRUE ~ col_space
      ),
      xwidth = dplyr::case_when(
        overlay & width < 0 ~ width - xsep,
        overlay ~ -xsep,
        TRUE ~ width
      ),
      xmax = cumsum(xwidth + xsep),
      xmin = xmax - xwidth,
      x = xmin + xwidth / 2
    )
  
  # Check metric columns exist (ignore the "Method" pseudo-column)
  need_ids <- setdiff(column_info$id, "Method")
  missing <- setdiff(need_ids, colnames(data))
  if (length(missing)) stop("Missing columns in `data`: ", paste(missing, collapse = ", "))
  
  # ---------- Method column (draw gray cell + text) ----------
  method_idx <- which(column_info$geom == "method" | column_info$id == "Method")
  method_labels <- tibble()
  if (length(method_idx) == 1) {
    cxmin <- column_pos$xmin[method_idx]
    # small left padding inside the method column
    method_labels <- tibble(
      x = cxmin + 0.10, 
      y = row_pos$y,
      label_value = row_info$id,
      hjust = 0, vjust = 0.5, size = 4, fontface = "bold",
      colors = "black", angle = 0
    )
  }
  
  # ---------- Circles ----------
  ind_circle <- which(column_info$geom == "circle")
  circle_data <- tibble()
  if (length(ind_circle) > 0) {
    cols_circle <- column_info %>% dplyr::slice(ind_circle) %>% dplyr::pull(id)
    dat_mat <- as.matrix(data[, cols_circle, drop = FALSE])
    
    col_palette <- data.frame(
      metric = colnames(dat_mat),
      group  = column_info[match(colnames(dat_mat), column_info$id), "group", drop = TRUE]
    )
    col_palette$name_palette <- lapply(col_palette$group, function(x) palettes[[as.character(x)]])
    
    circle_data <- tibble(
      label = rep(colnames(dat_mat), each = nrow(dat_mat)),
      x0    = rep(column_pos$x[ind_circle], each = nrow(dat_mat)),
      y0    = rep(row_pos$y, times = length(ind_circle)),
      r     = (row_height/2) * as.vector(sqrt(dat_mat))
    )
    for (lab in unique(circle_data$label)) {
      i <- circle_data$label == lab
      circle_data$r[i] <- scales::rescale(circle_data$r[i], to = c(0.05, 0.55),
                                          from = range(circle_data$r[i], na.rm = TRUE))
    }
    colors <- NULL
    for (i in seq_len(ncol(dat_mat))) {
      n_ok   <- sum(!is.na(dat_mat[, i]))
      base_p <- colorRampPalette(get_brewer(col_palette$name_palette[[i]], 9))(max(1, n_ok))
      colors <- c(colors, rev(base_p)[rank(dat_mat[, i], ties.method = "average", na.last = "keep")])
    }
    circle_data$colors <- colors
  }
  
  # ---------- Bars ----------
  ind_bar <- which(column_info$geom == "bar")
  rect_data <- tibble()
  if (length(ind_bar) > 0) {
    cols_bar <- column_info %>% dplyr::slice(ind_bar) %>% dplyr::pull(id)
    dat_mat  <- as.matrix(data[, cols_bar, drop = FALSE])
    
    if (rect_normalize) dat_mat <- apply(dat_mat, 2, function(x) x / max(x, na.rm = TRUE))
    
    col_palette <- data.frame(
      metric = colnames(dat_mat),
      group  = column_info[match(colnames(dat_mat), column_info$id), "group", drop = TRUE]
    )
    col_palette$name_palette <- lapply(col_palette$group, function(x) palettes[[as.character(x)]])
    
    rect_data <- tibble(
      label  = rep(colnames(dat_mat), each = nrow(dat_mat)),
      method = rep(row_info$id, times = ncol(dat_mat)),
      value  = as.vector(dat_mat),
      xmin   = rep(column_pos$xmin[ind_bar], each = nrow(dat_mat)),
      xmax   = rep(column_pos$xmax[ind_bar], each = nrow(dat_mat)),
      ymin   = rep(row_pos$ymin, times = ncol(dat_mat)),
      ymax   = rep(row_pos$ymax, times = ncol(dat_mat)),
      xwidth = rep(column_pos$xwidth[ind_bar], each = nrow(dat_mat))
    ) %>%
      add_column_if_missing(hjust = 0) %>%
      dplyr::mutate(
        xmin = xmin + (1 - value) * xwidth * hjust,
        xmax = xmax - (1 - value) * xwidth * (1 - hjust)
      )
    
    colors <- NULL
    for (i in seq_len(ncol(dat_mat))) {
      n_ok   <- sum(!is.na(dat_mat[, i]))
      base_p <- colorRampPalette(get_brewer(col_palette$name_palette[[i]], 9))(max(1, n_ok))
      colors <- c(colors, rev(base_p)[rank(dat_mat[, i], ties.method = "average", na.last = "keep")])
    }
    rect_data$colors <- colors
  }
  
  # ---------- Column headers (exclude Method) ----------
  df_hdr <- column_pos %>%
    dplyr::filter(id != "Method") %>%
    dplyr::mutate(
      header_dataset = dplyr::case_when(
        grepl("__Open$", id)     ~ "Open-ST",
        grepl("__Stereo$", id)   ~ "Stereo-seq",
        grepl("__MERFISH$", id)  ~ "MERFISH",
        TRUE                     ~ NA_character_
      ),
      header_metric = sub("(_|__)(Open|Stereo|MERFISH)$", "", id)
      #header_metric = sub("__(Open|Stereo|MERFISH)$", "", id)
    )
  
  
  text_data <- tibble()
  if (nrow(df_hdr) > 0) {
    
    # ticks
    segment_data <- dplyr::bind_rows(
      segment_data,
      df_hdr %>% dplyr::transmute(
        x = x, xend = x, y = -.3, yend = -.1,
        size = .5, colour = "black", linetype = "solid"
      )
    )
    
    header_gap <- 0.5
    
    # Dataset label
    text_data <- dplyr::bind_rows(
      text_data,
      df_hdr %>% dplyr::transmute(
        xmin = x, xmax = x, ymin = 0.35 + header_gap, ymax = 0.05 + header_gap,
        angle = 20, vjust = 0, hjust = 0,
        label_value = header_dataset,
        size = 3, colors = "black", fontface = "bold"
      )
    )
    
    # Metric label
    text_data <- dplyr::bind_rows(
      text_data,
      df_hdr %>% dplyr::transmute(
        xmin = x, xmax = x, ymin = 0, ymax = -0.5,
        angle = 20, vjust = 0, hjust = 0,
        label_value = header_metric,
        size = 3, colors = "black", fontface = "plain"
      )
    )
  }
  
  
  # ---------- Dashed dividers between groups ----------
  dividers <- column_pos %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(x = max(xmax) + 0.3, .groups = "drop") %>%
    dplyr::arrange(x) %>% 
    dplyr::mutate(idx = dplyr::row_number()) %>%
    dplyr::filter(idx < dplyr::n()) %>%
    dplyr::transmute(
      x = x, xend = x,
      y = min(row_pos$ymin) - 0.4,
      yend = max(row_pos$ymax) + 0.4,
      size = 0.5, colour = "#BDBDBD", linetype = "dashed"
    )
  if (nrow(dividers) > 0) segment_data <- dplyr::bind_rows(segment_data, dividers)
  
  # ---------- Build plot ----------
  g <- ggplot() +
    coord_equal(expand = FALSE) +
    scale_alpha_identity() +
    scale_colour_identity() +
    scale_fill_identity() +
    scale_size_identity() +
    scale_linetype_identity() +
    cowplot::theme_nothing() +
    theme(text = element_text(family = "serif"))
  
  # Alternating row bands across the whole grid
  df_bg <- row_pos %>% dplyr::filter(colour_background)
  if (nrow(df_bg) > 0) {
    g <- g + geom_rect(
      aes(xmin = min(column_pos$xmin) - .25,
          xmax = max(column_pos$xmax) + .25,
          ymin = ymin - (row_space / 2),
          ymax = ymax + (row_space / 2)
      ),
      df_bg, fill = "#DDDDDD"
    )
  }
  
  # Method column cells + text
  if (nrow(method_labels) > 0) {
    g <- g + geom_text(
      aes(x = x, y = y, label = label_value,
          colour = colors, hjust = hjust, vjust = vjust,
          size = size, fontface = fontface, angle = angle),
      data = method_labels
    )
  }
  
  # Circles
  if (nrow(circle_data) > 0) {
    g <- g + ggforce::geom_circle(aes(x0 = x0, y0 = y0, r = r, fill = colors),
                                  circle_data, size = .25)
  }
  
  # Bars
  if (nrow(rect_data) > 0) {
    rect_data <- rect_data %>%
      add_column_if_missing(alpha = 1, border = TRUE, border_colour = "black") %>%
      dplyr::mutate(border_colour = ifelse(border, border_colour, NA))
    g <- g + geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
          fill = colors, colour = border_colour, alpha = alpha),
      rect_data, linewidth = .25
    )
  }
  
  # Column headers
  if (nrow(text_data) > 0) {
    text_data <- text_data %>%
      add_column_if_missing(
        hjust = .5, vjust = .5, size = 3, fontface = "plain",
        colors = "black", lineheight = 1, angle = 0
      ) %>%
      dplyr::mutate(
        angle2 = angle / 360 * 2 * pi,
        cosa = round(cos(angle2), 2),
        sina = round(sin(angle2), 2),
        alphax = ifelse(cosa < 0, 1 - hjust, hjust) * abs(cosa) +
          ifelse(sina > 0, 1 - vjust, vjust) * abs(sina),
        alphay = ifelse(sina < 0, 1 - hjust, hjust) * abs(sina) +
          ifelse(cosa < 0, 1 - vjust, vjust) * abs(cosa),
        x = (1 - alphax) * xmin + alphax * xmax,
        y = (1 - alphay) * ymin + alphay * ymax
      ) %>%
      dplyr::filter(label_value != "")
    g <- g + geom_text(aes(x = x, y = y, label = label_value,
                           colour = colors, hjust = hjust, vjust = vjust,
                           size = size, fontface = fontface, angle = angle),
                       data = text_data)
  }
  
  # Dividers / ticks
  if (nrow(segment_data) > 0) {
    segment_data <- segment_data %>%
      add_column_if_missing(size = .5, colour = "black", linetype = "solid")
    g <- g + geom_segment(
      aes(x = x, xend = xend, y = y, yend = yend,
          size = size, colour = colour, linetype = linetype),
      segment_data
    )
  }
  
  # Margins
  minimum_x <- min(column_pos$xmin) - 0.5
  maximum_x <- max(column_pos$xmax) + 3
  minimum_y <- min(row_pos$ymin) - 2.5
  maximum_y <- max(row_pos$ymax) + 4.5
  if (extend_figure) { maximum_x <- maximum_x + 5; maximum_y <- maximum_y + 10 }
  
  g + expand_limits(x = c(minimum_x, maximum_x), y = c(minimum_y, maximum_y))
}

#methods <- c("Seurat","CARD","cell2location","GraphST","DestVI","RCTD","scVI","SingleR","spatialDWLS","spatialID","spotlight","tacco","Tangram")
methods <- c('BANKSY','SpaGCN','STdeconvolve')
# Human lymph node metrics

# shift to remove negative values

df_stereo <- read.csv('/Users/yuling_zhu/Downloads/unsupervisedMethods/development_metrics.csv')
df_open <- read.csv('/Users/yuling_zhu/Downloads/unsupervisedMethods/Human_metrics.csv')
df_MERFISH <- read.csv('/Users/yuling_zhu/Downloads/unsupervisedMethods/MouseSpinal_metrics.csv')
df_xenium <- read.csv('/Users/yuling_zhu/Downloads/unsupervisedMethods/Kidney_metrics.csv')
df_MERFISH <- df_MERFISH %>%
  dplyr::rename(`Peak Memory` = Memory)
df_stereo <- df_stereo %>%
  dplyr::rename(`Peak Memory` = Memory)
df_open <- df_open %>%
  dplyr::rename(`Peak Memory` = Memory)
df_xenium <- df_xenium %>%
  dplyr::rename(`Peak Memory` = Memory)
########
df_MERFISH <- df_MERFISH %>%
  dplyr::rename(`AvgBIO` = avgbio)
df_stereo <- df_stereo %>%
  dplyr::rename(`AvgBIO` = avgbio)
df_open <- df_open %>%
  dplyr::rename(`AvgBIO` = avgbio)
df_xenium <- df_xenium %>%
  dplyr::rename(`AvgBIO` = avgbio)

# combine datasets
metric_cols <- colnames(df_stereo)[-1]

df_merfish2   <- df_MERFISH   %>% dplyr::rename_with(~ paste0(.x, "__MERFISH"),   all_of(metric_cols))
df_open2   <- df_open   %>% dplyr::rename_with(~ paste0(.x, "__Open"),   all_of(metric_cols))
df_stereo2 <- df_stereo %>% dplyr::rename_with(~ paste0(.x, "__Stereo"), all_of(metric_cols))
df_xenium2 <- df_xenium %>% dplyr::rename_with(~ paste0(.x, "__Xenium"), all_of(metric_cols))

df_combo <- dplyr::bind_cols(df_open2, df_stereo2)
df_combo2 <- dplyr::bind_cols(df_combo, df_merfish2)
df_combo3 <- dplyr::bind_cols(df_combo2, df_xenium2)
row_info <- tibble(
  id    = methods,
  group = ""
)

# Add a first column called "Method" with geom = "method"
column_info_base <- tribble(
  ~id,       ~group,                 ~geom,     ~overlay, ~width,
  "Method",  "",                      "method",  FALSE,    5,   
  "ECS",    "Alignment Accuracy",     "circle",  FALSE,    1.1,
  "ASW",    "Alignment Accuracy",    "circle",     FALSE,    1.1,
  "AMI",    "Alignment Accuracy",    "circle",     FALSE,    1.1,
  "ARI",     "Alignment Accuracy",    "circle",  FALSE,    1.1,
  "NMI",     "Alignment Accuracy",    "circle",  FALSE,    1.1,
  "AvgBIO",     "Alignment Accuracy",    "bar",  FALSE,    1.1,
  
  "Time",      "Efficiency",           "circle",   FALSE,   1.1,
  "Peak Memory", "Efficiency",         "bar",   FALSE,   1.1
)
col_method  <- column_info_base %>% dplyr::filter(id == "Method")
col_metrics <- column_info_base %>% dplyr::filter(id != "Method")

column_info <- dplyr::bind_rows(
  col_method,
  col_metrics %>%
    dplyr::mutate(metric_id = id) %>%
    tidyr::crossing(dataset = c("Open", "Stereo", "MERFISH","Xenium")) %>%
    dplyr::mutate(
      id = paste0(metric_id, "__", dataset),
      header_metric = metric_id,
      header_dataset = dplyr::case_when(
        dataset == "Open"     ~ "Open-ST",
        dataset == "Stereo"   ~ "Stereo-seq",
        dataset == "MERFISH"  ~ "MERFISH",
        dataset == "Xenium"  ~ "Xenium",
        TRUE                  ~ NA_character_
      )
    ) %>%
    dplyr::select(-metric_id)
)

# Ensure correct column order
group_order   <- c("Alignment Accuracy", "Efficiency")
dataset_order <- c("Open", "Stereo", "MERFISH","Xenium")

column_info <- column_info %>%
  dplyr::mutate(
    dataset = dplyr::case_when(
      id == "Method"           ~ "Method",
      grepl("__Open$", id)     ~ "Open",
      grepl("__Stereo$", id)   ~ "Stereo",
      grepl("__MERFISH$", id)  ~ "MERFISH",
      grepl("__Xenium$", id)  ~ "Xenium",
      TRUE                     ~ NA_character_
    ),
    metric_base = dplyr::case_when(
      id == "Method" ~ "Method",
      TRUE ~ sub("__(Open|Stereo|MERFISH|Xenium)$", "", id)
    )
  ) %>%
  dplyr::arrange(
    factor(group, levels = c("", group_order)),
    factor(metric_base, levels = unique(col_metrics$id)),
    factor(dataset, levels = c("Method", dataset_order))
  ) %>%
  dplyr::select(-dataset, -metric_base)


palettes <- list(
  "Alignment Accuracy" = "Blues",
  "Efficiency" = "Purples"
)

g <- plot_scmmib_summary(
  data           = df_combo3,
  row_info       = row_info,
  column_info    = column_info,
  palettes       = palettes,
  rank           = TRUE,
  rect_normalize = TRUE,   # changed to account for large time/memory values
  extend_figure  = FALSE
)
print(g)
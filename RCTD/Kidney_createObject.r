library(spacexr)
library(Matrix)
##################################### csv to rds of ref data
### conda activate /home/huy21/anaconda3/envs/bindSC_R
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_ref"
query <- c('L', 'R') 
time_point <-  c('Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6')
for ( i in 1:length(time_point)){
    counts <- read.csv(file.path(refdir, paste0(time_point[i],"_counts.csv")), 
                header = TRUE, 
                row.names = 1, 
                colClasses = "character", 
                stringsAsFactors = FALSE)
    counts_t <- t(counts)
    # Step 3: Convert all values to integer
    storage.mode(counts_t) <- "integer"  # works on matrix only
    # Now you have: counts as integer matrix, row/col names preserved
    counts <- counts_t
    # counts <- t(read.csv(file.path(refdir, "counts.csv"), row.names = 1, check.names = FALSE)) # load in counts matrix
    # meta_data <- read.csv(file.path(refdir,"meta_data.csv"), row.names = 1, check.names = FALSE) # load in meta_data (barcodes, clusters, and nUMI)
    # load meta data of reference single cell 
    meta_data <- read.table(file.path(refdir, paste0(time_point[i],"meta_data.csv")), 
                            header = TRUE, 
                            row.names = 1, 
                            sep = ",", 
                            colClasses = "character", 
                            stringsAsFactors = FALSE)
    cell_types <- meta_data$celltype
    names(cell_types) <- rownames(meta_data) # create cell_types named list
    cell_types <- as.factor(cell_types) # convert to factor data type
    # nUMI <- as.numeric(meta_data$nUMI)
    # names(nUMI) <- rownames(meta_data)

    ### Create the Reference object
    #levels(cell_types) <- gsub("/", "_", levels(cell_types))
    reference <- Reference(counts, cell_types, min_UMI= 1)
    saveRDS(reference, file.path(refdir,paste0(time_point[i],'SCRef.rds')))
}
for ( i in 1:length(time_point)){
    for (j in 1:length(query)){
        datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/Kidney_input"
# Step 1: Read everything as character to avoid numeric coercion
        raw <- read.csv(file.path(datadir, paste0(time_point[i], query[j], "_counts.csv")), 
                        header = TRUE, 
                        row.names = 1, 
                        colClasses = "character", 
                        stringsAsFactors = FALSE)
        # Step 2: Transpose (genes as rows, barcodes as columns, or vice versa)
        raw_t <- t(raw)
        # Step 3: Convert all values to integer
        storage.mode(raw_t) <- "integer"  # works on matrix only
        # Now you have: counts as integer matrix, row/col names preserved
        counts <- raw_t
        # Step 1: Read all as character to preserve long barcodes
        coords_raw <- read.table(file.path(datadir, paste0(time_point[i], query[j], "_location.csv")), 
                                header = TRUE, 
                                row.names = 1, 
                                sep = ",", 
                                colClasses = "character", 
                                stringsAsFactors = FALSE)

        # Step 2: Convert all columns to numeric (e.g., x, y)
        coords <- data.frame(lapply(coords_raw, as.numeric), row.names = rownames(coords_raw))

        mSC <- SpatialRNA(coords, counts)
        saveRDS(mSC, file.path(datadir,paste0(time_point[i], query[j],'SCRaw.rds')))
        }
}
 
library(spacexr)
library(Matrix)
##################################### csv to rds of ref data
### Load in/preprocess your data, this might vary based on your file type
refdir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/HumanLymph_ref"
# load counts file which saved from adata.X
counts <- read.csv(file.path(refdir, "counts.csv"), 
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
meta_data <- read.table(file.path(refdir, "meta_data.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

cell_types <- meta_data$original_clusters
names(cell_types) <- rownames(meta_data) # create cell_types named list
cell_types <- as.factor(cell_types) # convert to factor data type

# nUMI <- as.numeric(meta_data$nUMI)
# names(nUMI) <- rownames(meta_data)

### Create the Reference object
#levels(cell_types) <- gsub("/", "_", levels(cell_types))
reference <- Reference(counts, cell_types, min_UMI=20)
saveRDS(reference, file.path(refdir,'SCRef.rds'))

##################################### csv to rds of ref data finished


##################################### csv to rds of raw data (query spatial data)
datadir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/RCTD/HumanLymph_input"
# Step 1: Read everything as character to avoid numeric coercion
raw <- read.csv(file.path(datadir, "counts.csv"), 
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
coords_raw <- read.table(file.path(datadir, "location.csv"), 
                         header = TRUE, 
                         row.names = 1, 
                         sep = ",", 
                         colClasses = "character", 
                         stringsAsFactors = FALSE)

# Step 2: Convert all columns to numeric (e.g., x, y)
coords <- data.frame(lapply(coords_raw, as.numeric), row.names = rownames(coords_raw))

# rownames(coords) <- coords$barcodes; coords$barcodes <- NULL # Move barcodes to rownames
# nUMI <- colSums(counts) # In this case, total counts per pixel is nUMI

### Create SpatialRNA object
mSC <- SpatialRNA(coords, counts)

saveRDS(mSC, file.path(datadir,'SCRaw.rds'))
 
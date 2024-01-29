# This is a script made for merging both table from the script
# & the table containing firing patterns from patch clamp recording

# Load the needed libraries
library('tidyverse')

# Create Dataframes from each table
pattern <- read.csv('../table/cell_stats.csv')
morpho <- read.csv('../build/morpho.csv')

# Import information from gracula

# Create a brand new table from both and export
data <- morpho %>% inner_join(pattern, by=join_by(image), copy=FALSE, keep=FALSE)
write.csv(data, "../build/combined_extract.csv", row.names=FALSE)
write.csv(data, "../output/combined_extract.csv", row.names=FALSE)
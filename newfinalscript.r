
library(GEOquery)

gset2 <- getGEO("GSE10072", GSEMatrix = TRUE)
expr2  <- exprs(gset2[[1]])
expr2  <- log2(expr2 + 1)

pdata2 <- pData(gset2[[1]])

print(colnames(pdata2))

print(head(pdata2$characteristics_ch1))
print(head(pdata2$source_name_ch1))
print(head(pdata2$title))
library(GEOquery)

# Download GSE10072
gset2  <- getGEO("GSE10072", GSEMatrix = TRUE)
expr2  <- exprs(gset2[[1]])
expr2  <- log2(expr2 + 1)
pdata2 <- pData(gset2[[1]])

# Show column names
print(colnames(pdata2))

print(head(pdata2$source_name_ch1))
print(table(pdata2$source_name_ch1))

# Get your top 100 genes from original analysis
top100_probes <- rownames(results)

# Check common probes between both datasets
common_probes <- intersect(top100_probes, rownames(expr2))
print(paste("Common probes found:", length(common_probes)))

# Extract those genes from GSE10072
expr2_subset     <- expr2[common_probes, ]
expr2_transposed <- t(expr2_subset)
validation_data  <- as.data.frame(expr2_transposed)

# Add correct labels
validation_data$condition <- ifelse(
  pdata2$source_name_ch1 == "Adenocarcinoma of the Lung",
  "Tumor", "Normal"
)

# Check label distribution
print(table(validation_data$condition))

# Save
write.csv(validation_data, 
          "C:/Users/bhaav/OneDrive/Desktop/aiml/validation_data.csv", 
          row.names = FALSE)

print(paste("Saved validation_data.csv with", nrow(validation_data), "patients"))
print(paste("Features:", ncol(validation_data) - 1, "genes"))
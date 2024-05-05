library(dplyr)
library(data.table)

setwd("~/Desktop/mit/brand_generic")
load("src/hemonc_processing/drugs.RData")
load("src/hemonc_processing/indications.RData")

a <- merge(indications, drugs, by.x= "component", by.y = "drug")

data.table::fwrite(a, "hemonc_joined.csv")



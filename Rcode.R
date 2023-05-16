library(readr)
library(dplyr)

fd = read_csv(url("https://raw.githubusercontent.com/merry555/FibVID/main/claim_propagation/claim_propagation.csv"))

fdc = fd[c(3,8,9,11)]
dataset = filter(fdc, group %in%  c(0,1))


dataset =na.omit(dataset)



write.csv(dataset, "news_dataset.csv")
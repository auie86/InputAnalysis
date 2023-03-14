library("tidyverse")
library("fitdistrplus")


mydata <- read_csv("data/transactions.csv")

df <- rename(filter(mydata, Client == 20), x=Fee)
# some descriptive statistics
mean(df$x)
sd(df$x)

# Histogram
ggplot(data = df) +
  geom_histogram(mapping = aes(x = x)) +
  geom_vline(xintercept=mean(df$x), color="red") + 
  geom_vline(xintercept=median(df$x), color="blue") 

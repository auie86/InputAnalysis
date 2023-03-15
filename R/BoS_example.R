#
# Dist Fitting example with a large data set of currency swap transactions.
#
# 2023-03-14 - Jeff Smith
#
#
library("tidyverse")
library("fitdistrplus")
library("mc2d")

mydata <- read_csv("data/transactions.csv")

df <- rename(filter(mydata, Day == 1), x=Fee)
# some descriptive statistics
mean(df$x)
sd(df$x)

# Histogram
ggplot(data = df) +
  geom_histogram(mapping = aes(x = x)) +
  geom_vline(xintercept=mean(df$x), color="red") + 
  geom_vline(xintercept=median(df$x), color="blue") 

plotdist(df$x, histo = TRUE, demp = TRUE)

# normal
fn <- fitdist(df$x, "norm")
summary(fn)

# Weibull
fw <- fitdist(df$x, "weibull")
summary(fw)

# lognormal
fln <- fitdist(df$x, "lnorm")
summary(fln)

# Triangular
# From https://stackoverflow.com/questions/39981889/fit-triangular-distribution
Mode_fc <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
MyParam <- list(min= min(df$x)-.01, max= max(df$x)+.01, mode= Mode_fc(df$x))
ftr <- fitdist(df$x, "triang", method="mge", start = MyParam, gof="KS")  ## works
summary(ftr)

# compare
plot.legend <- c("Normal", "Weibull", "lognormal", "Triangular")
l <- list(fn, fw, fln, ftr)
par(mfrow = c(2, 2))
denscomp(l,legendtext = plot.legend)
qqcomp(l, legendtext = plot.legend)
cdfcomp(l, legendtext = plot.legend)      
ppcomp(l, legendtext = plot.legend)

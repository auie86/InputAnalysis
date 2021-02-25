#
# dist_fitting_example.R
#
# 2020-02-20 - Jeff Smith
#
library("tidyverse")
library("fitdistrplus")

# Read the dataset
days <- read_csv("../data/days_in_storage.csv")

# some descriptive statistics
mean(days$days)
sd(days$days)

# Histogram
ggplot(data = days) +
  geom_histogram(mapping = aes(x = days), binwidth=20) +
  geom_vline(xintercept=mean(days$days), color="red") + 
  geom_vline(xintercept=median(days$days), color="blue") 

# Distribution fitting
# from https://cran.r-project.org/web/packages/fitdistrplus/vignettes/paper2JSS.pdf
# Empirical PDF/CDF
plotdist(days$days, histo = TRUE, demp = TRUE)

descdist(days$days, boot = 1000)


# normal
fn <- fitdist(days$days, "norm")
summary(fn)

# Weibull
fw <- fitdist(days$days, "weibull")
summary(fw)

# lognormal
fln <- fitdist(days$days, "lnorm")
summary(fln)

# gamma
fg <- fitdist(days$days, "gamma")
summary(fg)

# compare
plot.legend <- c("Normal", "Weibull", "lognormal", "gamma")
par(mfrow = c(2, 2))
denscomp(list(fn, fw, fln, fg),legendtext = plot.legend)
qqcomp(list(fn, fw, fln, fg), legendtext = plot.legend)
cdfcomp(list(fn, fw, fln, fg), legendtext = plot.legend)      
ppcomp(list(fn, fw, fln, fg), legendtext = plot.legend)

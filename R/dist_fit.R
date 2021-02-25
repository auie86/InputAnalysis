#
# dist_fit.R
#
# 2020-02-20 - Jeff Smith
#
library("tidyverse")
library("fitdistrplus")

# Read the dataset
mydata <- read_csv("../data/days_in_storage.csv")

names(mydata)[1] <- "x"

# some descriptive statistics
mean(mydata$x)
sd(mydata$x)

# Histogram
ggplot(data = mydata) +
  geom_histogram(mapping = aes(x = x), binwidth=15) +
  geom_vline(xintercept=mean(mydata$x), color="red") + 
  geom_vline(xintercept=median(mydata$x), color="blue") 

# Distribution fitting
# from https://cran.r-project.org/web/packages/fitdistrplus/vignettes/paper2JSS.pdf
# Empirical PDF/CDF
plotdist(mydata$x, histo = TRUE, demp = TRUE)

#descdist(days$days, boot = 1000)


# normal
fn <- fitdist(mydata$x, "norm")
summary(fn)

# Weibull
fw <- fitdist(mydata$x, "weibull")
summary(fw)

# lognormal
fln <- fitdist(mydata$x, "lnorm")
summary(fln)

# gamma
fg <- fitdist(mydata$x, "gamma")
summary(fg)

# compare
plot.legend <- c("Normal", "Weibull", "lognormal", "gamma")
par(mfrow = c(2, 2))
denscomp(list(fn, fw, fln, fg),legendtext = plot.legend)
qqcomp(list(fn, fw, fln, fg), legendtext = plot.legend)
cdfcomp(list(fn, fw, fln, fg), legendtext = plot.legend)      
ppcomp(list(fn, fw, fln, fg), legendtext = plot.legend)

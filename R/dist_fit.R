#
# dist_fit.R
#
# 2020-02-20 - Jeff Smith
#
library("tidyverse")
library("fitdistrplus")
library("mc2d")

# Read the dataset
mydata <- read_csv("data/obs.csv")

# Rename the column
mydata <- rename(mydata, x=Obs)

# names(mydata)[1] <- "x"

# some descriptive statistics
mean(mydata$x)
sd(mydata$x)

# Histogram
ggplot(data = mydata) +
  geom_histogram(mapping = aes(x = x)) +
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

# Triangular
# From https://stackoverflow.com/questions/39981889/fit-triangular-distribution
Mode_fc <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
MyParam <- list(min= min(mydata$x)-.5, max= max(mydata$x)+.5, mode= Mode_fc(mydata$x))
ftr <- fitdist(mydata$x, "triang", method="mge", start = MyParam, gof="KS")  ## works
summary(ftr)


# compare
plot.legend <- c("Normal", "Weibull", "lognormal", "gamma", "Triangular")
l <- list(fn, fw, fln, fg, ftr)
par(mfrow = c(2, 2))
denscomp(l,legendtext = plot.legend)
qqcomp(l, legendtext = plot.legend)
cdfcomp(l, legendtext = plot.legend)      
ppcomp(l, legendtext = plot.legend)

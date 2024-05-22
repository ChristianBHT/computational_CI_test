library(xgboost)
library(ggplot2)
library(nortest)
library(goftest)
set.seed(84)
trans_red <- rgb(1, 0, 0, alpha = 0.5)
trans_blue <- rgb(0, 0, 1, alpha = 0.5)

data <- CategorizeInteractiondData(800)

output <- list()

for (j in 1:1000) {
  output[[j]] <- NullGenerator(data = data, formula = Y ~ X + Z2 + Z1, p = 0.85, objective = 'multi:softprob', num_class = 4)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}
NullDist <- data.frame(do.call(rbind, output))
Test <- TestGenerator(data = data, formula =  Y ~ X + Z2 + Z1, p = 0.85, objective = 'multi:softprob', num_class = 4)
test1 <- Test[1]
test2 <- Test[2]

get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

output2 <- list()
for (j in 1:1000) {
  output2[[j]] <- TestGenerator(data = data, formula =  Y ~ X + Z2 + Z1 , p = 0.85, objective = 'multi:softprob', num_class = 4)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}

TestDist <- data.frame(do.call(rbind, output2))


pdf("null_distributions.pdf", width = 6, height = 4)

par(mfrow = c(1, 2))

hist(NullDist$Metric1, col = trans_blue, xlim = c(min(c(TestDist$Metric1, NullDist$Metric1)),
                                                  max(c(TestDist$Metric1, NullDist$Metric1))),
     main = "", xlab = "Log loss", ylab = "Frequency", breaks = 15)

hist(NullDist$Metric2, col = trans_blue, xlim = c(min(c(TestDist$Metric2, NullDist$Metric2)),
                                                  max(c(TestDist$Metric2, NullDist$Metric2))),
     main = "", xlab = "Kappa scores", ylab = "Frequency", breaks = 10)

par(mfrow = c(1, 1))
dev.off()

pdf("overlay_distributions.pdf", width = 6, height = 4)

par(mfrow = c(1, 2))

hist(NullDist$Metric1, col = trans_blue, xlim = c(min(c(TestDist$Metric1, NullDist$Metric1)),
                                                  max(c(TestDist$Metric1, NullDist$Metric1))),
     main = "", xlab = "Log loss", ylab = "Frequency", breaks = 15)
hist(TestDist$Metric1, col = trans_red, add = T,breaks = 15)

hist(NullDist$Metric2, col = trans_blue, xlim = c(min(c(TestDist$Metric2, NullDist$Metric2)),
                                                  max(c(TestDist$Metric2, NullDist$Metric2))),
     main = "", xlab = "Kappa scores", ylab = "Frequency", breaks = 10)
hist(TestDist$Metric2, col = trans_red, breaks = 10, add = TRUE)

par(mfrow = c(1, 1))
dev.off()

p_values <- list()
for (i in 1:1000) {
  test1 <- TestDist$Metric1[i]
  test2 <- TestDist$Metric2[i]
  p_values[[i]] <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

}


pvalues <- data.frame(do.call(rbind, p_values))

pdf("pvalues_under_null.pdf", width = 6, height = 4)

par(mfrow = c(1, 2))

qqplot(qunif(ppoints(pvalues$p_value1)), pvalues$p_value1, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = " ")
abline(0, 1, col = "red")

qqplot(qunif(ppoints(pvalues$p_value2)), pvalues$p_value2, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = " ")
abline(0, 1, col = "red")

par(mfrow = c(1, 1))
dev.off()

#################### Untrue null distribution###############################

output <- list()

for (j in 1:1000) {
  output[[j]] <- NullGenerator(data = data, formula = Y ~ X + Z2, p = 0.85, objective = 'multi:softprob', num_class = 4)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}
NullDist <- data.frame(do.call(rbind, output))

output2 <- list()
for (j in 1:1000) {
  output2[[j]] <- TestGenerator(data = data, formula = Y ~ X + Z2 , p = 0.85, objective = 'multi:softprob', num_class = 4)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}

TestDist <- data.frame(do.call(rbind, output2))
test1 <- TestDist$Metric1[1]
test2 <- TestDist$Metric2[1]
get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

pdf("overlay_distributions_false_null.pdf", width = 6, height = 4)

par(mfrow = c(1, 2))

hist(NullDist$Metric1, col = trans_blue, xlim = c(min(c(TestDist$Metric1, NullDist$Metric1)),
                                                  max(c(TestDist$Metric1, NullDist$Metric1))),
     main = "", xlab = "Log loss", ylab = "Frequency", breaks = 25)
hist(TestDist$Metric1, col = trans_red, add = T,breaks = 25)

hist(NullDist$Metric2, col = trans_blue, xlim = c(min(c(TestDist$Metric2, NullDist$Metric2)),
                                                  max(c(TestDist$Metric2, NullDist$Metric2))),
     main = "", xlab = "Kappa scores", ylab = "Frequency", breaks = 25)
hist(TestDist$Metric2, col = trans_red, breaks = 35, add = TRUE)

par(mfrow = c(1, 1))
dev.off()

p_values <- list()
for (i in 1:1000) {
  test1 <- TestDist$Metric1[i]
  test2 <- TestDist$Metric2[i]
  p_values[[i]] <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

}


pvalues <- data.frame(do.call(rbind, p_values))


pdf("pvalues_false_null.pdf", width = 6, height = 4)

par(mfrow = c(1, 2))

qqplot(qunif(ppoints(pvalues$p_value1)), pvalues$p_value1, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = "P-values from log loss")
abline(0, 1, col = "red")

qqplot(qunif(ppoints(pvalues$p_value2)), pvalues$p_value2, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = "P-values from log loss")
abline(0, 1, col = "red")

par(mfrow = c(1, 1))
dev.off()


################# Simulation with low power ###################

data <- multinominal(800, zeta = 1)
data$X <- as.factor(data$X)
data$Y <- as.integer(as.factor(data$Y))-1


output <- list()

for (j in 1:1000) {
  output[[j]] <- NullGenerator(data = data, formula = Y ~ X + Z2, p = 0.85, objective = 'multi:softprob', num_class = 3)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}

NullDist <- data.frame(do.call(rbind, output))

test <- list()
for (j in 1:1000) {
  test[[j]] <- TestGenerator(data = data, formula =  Y ~ X + Z2, p = 0.85, objective = 'multi:softprob', num_class = 3)
  cat(sprintf("Sample: %d\r", j))
  flush.console()
}

TestDist <- data.frame(do.call(rbind, test))

pdf("overlay_distrutions_low_power.pdf", width = 6, height = 4)
par(mfrow = c(1, 2))
hist(NullDist$Metric1, col = trans_blue, ylim = range(0,300), xlim = c(min(c(TestDist$Metric1, NullDist$Metric1)),
                                                                       max(c(TestDist$Metric1, NullDist$Metric1))),
     main = "", xlab = "Log loss", ylab = "Frequency", breaks = 15)
hist(TestDist$Metric1, col = trans_red, add = T,breaks = 15)

hist(NullDist$Metric2, col = trans_blue, ylim = range(0,300), xlim = c(min(c(TestDist$Metric2, NullDist$Metric2)),
                                                                       max(c(TestDist$Metric2, NullDist$Metric2))),
     main = "", xlab = "Kappa scores", ylab = "Frequency", breaks = 15)
hist(TestDist$Metric2, col = trans_red, breaks = 15, add = TRUE)
par(mfrow = c(1, 1))
dev.off()

p_values <- list()
for (i in 1:1000) {
  test1 <- TestDist$Metric1[i]
  test2 <- TestDist$Metric2[i]
  p_values[[i]] <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

}
pvalues <- data.frame(do.call(rbind, p_values))

pdf("qq_plots_low_power.pdf", width = 6, height = 4)
par(mfrow = c(1, 2))
qqplot(qunif(ppoints(pvalues$p_value1)), pvalues$p_value1, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = "P-values from log loss")
abline(0, 1, col = "red")
qqplot(qunif(ppoints(pvalues$p_value2)), pvalues$p_value2, xlab = "Theoretical Quantiles", ylab = "Sample Quantiles", main = "P-values from Kappa scores")
abline(0, 1, col = "red")
par(mfrow = c(1, 1))
dev.off()

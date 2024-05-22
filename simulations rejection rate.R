library(parallel)
library(pbapply)
library(dplyr)
library(xgboost)
library(caret)
library(Metrics)
library(diptest)

cl <- makeCluster(detectCores() - 1)
for (N in c(500, 800, 2000)) {
  p <- 0.825
  R <- 1000
  no_tests <- 100

  clusterExport(cl, varlist = c('NullGenerator',
                                'TestGenerator',
                                'multi_class_log_loss',
                                'get_pvalues',
                                'CategorizeInteractiondData',
                                'simulateExpLogData',
                                'simulateTrigData',
                                'simulatePolyData',
                                'multinominal',
                                'N',
                                'R',
                                'p',
                                'no_tests'), envir = environment())

  clusterEvalQ(cl, {
    library(caret)
    library(Metrics)
    library(dplyr)
    library(xgboost)
    library(diptest)
  })

  results <- pblapply(cl=cl, 1:no_tests, function(i){

    Interactiondata <- CategorizeInteractiondData(N)
    Interactiondata$Z2Sqr <- Interactiondata$Z2^2
    Interactiondata$Z2Cub <- Interactiondata$Z2^3
    Interactiondata$sinZ2 <- sin(Interactiondata$Z2)
    Interactiondata$cosZ2 <- cos(Interactiondata$Z2)
    Interactiondata$expZ2 <- exp(Interactiondata$Z2)
    Interactiondata$logZ2 <- exp(abs(Interactiondata$Z2))
    Interactiondata$X <- as.factor(Interactiondata$X)
    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = Interactiondata,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 4)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = Interactiondata,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 4)
    test1 <- test[1]
    test2 <- test[2]

    InteractionP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

    ################

    EpxLogData <- simulateExpLogData(N)

    EpxLogData$Z2Sqr <- EpxLogData$Z2^2

    EpxLogData$Z2Cub <- EpxLogData$Z2^3

    EpxLogData$sinZ2 <- sin(EpxLogData$Z2)

    EpxLogData$cosZ2 <- cos(EpxLogData$Z2)

    EpxLogData$expZ2 <- exp(EpxLogData$Z2)

    EpxLogData$logZ2 <- exp(abs(EpxLogData$Z2))
    EpxLogData$Y <- EpxLogData$Y - 1

    EpxLogData$X <- as.factor(EpxLogData$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = EpxLogData,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 3)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = EpxLogData,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 3)
    test1 <- test[1]
    test2 <- test[2]

    ExpLogP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)
    ExpLogP_values

    ################

    TrigData <- simulateTrigData(N)

    TrigData$Z2Sqr <- TrigData$Z2^2

    TrigData$Z2Cub <- TrigData$Z2^3

    TrigData$sinZ2 <- sin(TrigData$Z2)

    TrigData$cosZ2 <- cos(TrigData$Z2)

    TrigData$expZ2 <- exp(TrigData$Z2)

    TrigData$logZ2 <- exp(abs(TrigData$Z2))
    TrigData$Y <- TrigData$Y - 1
    TrigData$X <- as.factor(TrigData$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = TrigData,
                                   formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 3)
      # cat(sprintf("Sample: %d\r", j))
      # flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = TrigData,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 3)
    test1 <- test[1]
    test2 <- test[2]

    TrigDataP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

    #################

    PolyData <- simulatePolyData(N)
    PolyData$Z2Sqr <- PolyData$Z2^2
    PolyData$Z2Cub <- PolyData$Z2^3
    PolyData$sinZ2 <- sin(PolyData$Z2)
    PolyData$cosZ2 <- cos(PolyData$Z2)
    PolyData$expZ2 <- exp(PolyData$Z2)
    PolyData$logZ2 <- exp(abs(PolyData$Z2))
    PolyData$X <- as.factor(PolyData$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = PolyData,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 4)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = PolyData,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 4)
    test1 <- test[1]
    test2 <- test[2]

    PolyDataP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

    ###################

    NonLinearData <- simulateNonLinearData(N)
    NonLinearData$Z1Sqr <- NonLinearData$Z1^2
    NonLinearData$Z2Sqr <- NonLinearData$Z2^2
    NonLinearData$Z1Cub <- NonLinearData$Z1^3
    NonLinearData$Z2Cub <- NonLinearData$Z2^3
    NonLinearData$sinZ1 <- sin(NonLinearData$Z1)
    NonLinearData$sinZ2 <- sin(NonLinearData$Z2)
    NonLinearData$cosZ1 <- cos(NonLinearData$Z1)
    NonLinearData$cosZ2 <- cos(NonLinearData$Z2)
    NonLinearData$expZ1 <- exp(NonLinearData$Z1)
    NonLinearData$expZ2 <- exp(NonLinearData$Z2)
    NonLinearData$logZ1 <- exp(abs(NonLinearData$Z1))
    NonLinearData$logZ2 <- exp(abs(NonLinearData$Z2))
    NonLinearData$X <- as.factor(NonLinearData$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = NonLinearData,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 4)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = NonLinearData,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 4)
    test1 <- test[1]
    test2 <- test[2]

    NonLinearDataP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

    ################
    ComplexCategorization <- simulateComplexCategorization(N)
    ComplexCategorization$Z1Sqr <- ComplexCategorization$Z1^2
    ComplexCategorization$Z2Sqr <- ComplexCategorization$Z2^2
    ComplexCategorization$Z1Cub <- ComplexCategorization$Z1^3
    ComplexCategorization$Z2Cub <- ComplexCategorization$Z2^3
    ComplexCategorization$sinZ1 <- sin(ComplexCategorization$Z1)
    ComplexCategorization$sinZ2 <- sin(ComplexCategorization$Z2)
    ComplexCategorization$cosZ1 <- cos(ComplexCategorization$Z1)
    ComplexCategorization$cosZ2 <- cos(ComplexCategorization$Z2)
    ComplexCategorization$expZ1 <- exp(ComplexCategorization$Z1)
    ComplexCategorization$expZ2 <- exp(ComplexCategorization$Z2)
    ComplexCategorization$logZ1 <- exp(abs(ComplexCategorization$Z1))
    ComplexCategorization$logZ2 <- exp(abs(ComplexCategorization$Z2))
    ComplexCategorization$X <- as.factor(ComplexCategorization$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = ComplexCategorization,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 4)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = ComplexCategorization,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 4)
    test1 <- test[1]
    test2 <- test[2]

    ComplexCategorizationP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)
    ################

    mlogData <- multinominal(N)
    mlogData$Z1Sqr <- mlogData$Z1^2
    mlogData$Z2Sqr <- mlogData$Z2^2
    mlogData$Z1Cub <- mlogData$Z1^3
    mlogData$Z2Cub <- mlogData$Z2^3
    mlogData$sinZ1 <- sin(mlogData$Z1)
    mlogData$sinZ2 <- sin(mlogData$Z2)
    mlogData$cosZ1 <- cos(mlogData$Z1)
    mlogData$cosZ2 <- cos(mlogData$Z2)
    mlogData$expZ1 <- exp(mlogData$Z1)
    mlogData$expZ2 <- exp(mlogData$Z2)
    mlogData$logZ1 <- exp(abs(mlogData$Z1))
    mlogData$logZ2 <- exp(abs(mlogData$Z2))
    mlogData$Y <- as.integer(as.factor(mlogData$Y))-1
    mlogData$X <- as.factor(mlogData$X)

    output <- list()
    for (j in 1:R) {
      output[[j]] <- NullGenerator(data = mlogData,
                                   formula =  Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                                   p = p,
                                   objective = 'multi:softprob',
                                   num_class = 3)
      cat(sprintf("Sample: %d\r", j))
      flush.console()
    }

    NullDist <- data.frame(do.call(rbind, output))

    test <- TestGenerator(data = mlogData,
                          formula = Y ~ X + Z2 + Z2Sqr + Z2Cub + sinZ2 + cosZ2 + expZ2 + logZ2 ,
                          p = p,
                          objective = 'multi:softprob',
                          num_class = 3)
    test1 <- test[1]
    test2 <- test[2]

    mlogDataP_values <- get_pvalues(objective = 'multi:softprob', NullDist = NullDist, test1_metric = test1, test2_metric = test2)

    ###################

    return(data.frame(
      test_number = i,
      error = "rejection rate",
      CI_statement = "Y _||_ X | Z2 ",
      InteractionP_values1 = InteractionP_values[1],
      InteractionP_values2 = InteractionP_values[2],
      ExpLogP_values1 =  ExpLogP_values[1],
      ExpLogP_values2 =  ExpLogP_values[2],
      TrigDataP_values1 = TrigDataP_values[1],
      TrigDataP_values2 = TrigDataP_values[2],
      PolyDataP_values1 = PolyDataP_values[1],
      PolyDataP_values2 = PolyDataP_values[2],
      NonLinearDataP_values1 = NonLinearDataP_values[1],
      NonLinearDataP_values2 = NonLinearDataP_values[2],
      ComplexCategorizationP_values1 =  ComplexCategorizationP_values[1],
      ComplexCategorizationP_values2 =  ComplexCategorizationP_values[2],
      mlogDataP_values1 = mlogDataP_values[1],
      mlogDataP_values2 = mlogDataP_values[2]
    ))
  })

  # Save output
  filename <- paste0("simulation_rejection_rate_", N,".rds")
  saveRDS(results, filename)
}
stopCluster(cl)



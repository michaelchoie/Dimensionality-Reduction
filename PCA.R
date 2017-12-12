# Dataset from UC Irvine Machine Learning Repo
# https://archive.ics.uci.edu/ml/datasets/Gisette
# Dealing with high dimensional data via PCA & Genetic Algorithm

library(RCurl) # for composing HTML requests
library(pryr) # for measuring size of objects
library(data.table)
library(doParallel)
library(caret) # for reducing zero/near-zero variance, partitioning data, modeling
library(pROC) # for calculating AUC and plotting ROC Curve

# Load data
loadData <- function(urlfile) {
    x <- getURL(urlfile, ssl.verifypeer = FALSE)
    read.table(textConnection(x), sep = '', header = FALSE, stringsAsFactors = FALSE)
}

urlfileData <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data'
urlfileLabels <- "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels"

gisetteRaw <- loadData(urlfileData)
g_labels <- loadData(urlfileLabels)
object_size(gisetteRaw)
object_size(g_labels)

print(dim(gisetteRaw))

# Get rid of near-zero variance features as they provide little information & risk crashing the program
# Quick and dirty fix but not a best practice (as these features may contain important information)
nzv <- nearZeroVar(gisetteRaw, saveMetrics = TRUE)
print(paste("Range:", round(range(nzv$percentUnique), 2)))
print(head(nzv))

# Remove features with less than 0.1% variance
print(paste("Number of features before cleaning:", nrow(nzv)))
gisette_nzv <- gisetteRaw[c(rownames(nzv[nzv$percentUnique > 0.1,])) ]
print(paste("Number of features after cleaning:", ncol(gisette_nzv)))

# Make the dependent variable nominal
g_labels <- factor(ifelse(g_labels == 1, "Yes", "No"), levels = c("Yes", "No"))

# Bind the labels to the dataset
dfEvaluateOrig <- cbind(as.data.frame(sapply(gisette_nzv, as.numeric)),
                        cluster = g_labels)

# Define function to evaluate AUC using XGBoost
evaluateAUC <- function(dfEvaluateOrig) {

    set.seed(1)
    trainIndex <- createDataPartition(dfEvaluateOrig$cluster, p = 0.8, list = F, times = 1)
    dataTrain <- dfEvaluateOrig[trainIndex, ]
    dataTest <- dfEvaluateOrig[-trainIndex, ]

    controlParameters <- trainControl(method = "cv",
                                      number = 5, # number of folds
                                      verboseIter = FALSE, # logical for printing training log
                                      returnData = FALSE, # saves data to slot called trainingData
                                      returnResamp = "all", # save resampled performance measures
                                      classProbs = TRUE, # set to TRUE for class probabilities to be computed
                                      summaryFunction = twoClassSummary, # performance summaries
                                      allowParallel = TRUE) # should back end parallel processing clusters be used

    parametersGrid <-  expand.grid(nrounds = 10, # number of iterations the model runs
                                   eta = 0.3, # learning rate which is step size shrinkage which actually shrinks the feature weights
                                   gamma = 1, # minimum loss reduction required to make a further partition on a leaf node of the tree
                                   max_depth = 6, # how big of a tree to create
                                   min_child_weight= 1, # minimum Sum of Instance Weight
                                   colsample_bytree= 0.8, # randomly choosing the number of columns out of all columns during tree building process
                                   subsample = 1) # part of data instances to grow tree

    # Register parallel processing back end
    cl <- makeCluster(3)
    registerDoParallel(cl)

    # Train model
    xgBoost_Model <- train(cluster ~ ., data = dataTrain,
                           method = "xgbTree", metric = "ROC",
                           trControl = controlParameters,
                           tuneGrid = parametersGrid)

    # Close cluster
    stopCluster(cl)

    # Calculate probabilities
    predictions <- predict(xgBoost_Model, dataTest[, -ncol(dataTest)], type = "prob")

    # Evaluate performance
    print(auc(ifelse(dataTest[, "cluster"] == "Yes", 1, 0), predictions[[1]]))

}

# Create function to calculate how long a function takes to run
timeAlgorithm <- function(algorithm) {

    tStart <- Sys.time()
    algorithm
    tEnd <- Sys.time()
    difftime(tEnd, tStart, units = "secs")

}

# Evaluate on original dataset
timeAlgorithm(evaluateAUC(dfEvaluateOrig))

# Conduct feature scaling and PCA

# Standardize the data and find principal components
princ <- prcomp(gisette_nzv, scale = T)

# Determine how many principal components to choose given variance explained
pve <- princ$sdev^2 / sum(princ$sdev^2)
plot(cumsum(pve))

# Prepare data for evaluation
nComp <- 1000
dfComponents <- predict(princ, newdata = scale(gisette_nzv))[, 1:nComp]
dfEvaluatePCA <- cbind(as.data.frame(dfComponents), cluster = g_labels)

# Evaluate on principal component dataset
timeAlgorithm(evaluateAUC(dfEvaluatePCA))


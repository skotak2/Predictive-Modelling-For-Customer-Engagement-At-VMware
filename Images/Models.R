library (PivotalR) #deals with PostgreSQL or Pivotal databases
library (RPostgreSQL) #access PostgreSQL databases
library (DBI) #interface between R and relational DBMS
library (data.table)
library (randomForest)  # tree based model - bagging
library (nnet)  		# neural network
library (Matrix)
library (foreach) 
library (glmnet)		# linear model
library (brnn)
library (lattice)
library(ggplot2)
library (caret)
library (RRF)
library (dummies)
library (gbm)
library (xgboost)
library (LiblineaR)
library (nnls)

setwd ("")
read.csv("")

# Multiclass AUC
calc_auc <- function (actual, predicted)
{
  r <- rank(predicted)
  n_pos <- as.numeric (sum(actual == 1))
  n_neg <- as.numeric (length(actual) - n_pos)
  denom <- as.double (as.double (n_pos) * as.double(n_neg))
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(denom)
  auc
}
setDT(digitaltrain)
for (monkey in 1:5) {
  actual <- ifelse (digitaltrain$target==monkey,1,0)
  aucDF <- digitaltrain [,allnms, with = F][,lapply(.SD, function (x) calc_auc (actual, x))]
  aucDF <- as.data.frame (aucDF)
  aucDF <- t (aucDF)
  aucDF <- as.data.frame (aucDF)
  aucDF$varName <- rownames (aucDF)
  names (aucDF)[1] <- paste ("target", monkey, sep="_")
  if (monkey == 1) bigaucdf <- aucDF	
  if (monkey != 1) {
    bigaucdf <- merge (bigaucdf, aucDF, by = "varName", all.x=TRUE)
  }
}

bigaucdf$target_1 <- abs (bigaucdf$target_1 - 0.5)
bigaucdf$target_2 <- abs (bigaucdf$target_2 - 0.5)
bigaucdf$target_3 <- abs (bigaucdf$target_3 - 0.5)
bigaucdf$target_4 <- abs (bigaucdf$target_4 - 0.5)
bigaucdf$target_5 <- abs (bigaucdf$target_5 - 0.5)
bigaucdf$max_auc <- apply (bigaucdf, 1, function (x) as.numeric (max (x[2:6])))
write.csv (bigaucdf, file = 'bigauc_df.csv', row.names = F)


varSel_auc_0.001 <- bigaucdf$varName [which (bigaucdf$max_auc > 0.001)]
varSel_auc_0.007 <- bigaucdf$varName [which (bigaucdf$max_auc > 0.007)]
varSel_auc_0.03 <- bigaucdf$varName [which (bigaucdf$max_auc > 0.03)]
varSel_auc_0.01 <- bigaucdf$varName [which (bigaucdf$max_auc  > 0.01)]

lift_model <- function(actual,myPred,groups)
{
  actual <- actual [order(myPred, decreasing=TRUE)]
  myPred <- myPred [order (myPred, decreasing=TRUE)]
  deciles <- rep (1:groups, each = length(actual)/groups)
  deciles <- c(deciles, rep (groups, length (actual) - length(deciles)))
  naiveAcc <- prop.table (table (actual))[2]
  #naiveAcc  <- naiveAcc/2
  #tempDF <- data.frame (actual, deciles, myPred) 
  cumlifts <- NULL
  for (j in 1:groups) {
    cumlifts <- c(cumlifts, length (which (actual [deciles <= j]==1))/(naiveAcc * length (actual [deciles <= j])))
  }
  print (paste (cumlifts[1], cumlifts [5]))
}
digitaltrain <- as.data.frame (digitaltrain)
digitalvalid <- as.data.frame (digitalvalid)
digitaltrain$target <- ifelse (digitaltrain$target==-1,0,digitaltrain$target)
digitalvalid$target <- ifelse (digitalvalid$target==-1,0,digitalvalid$target)

save.image ('beforemodel.rData')

############################################################################
#########MODELING ENGINE###########################################
###############################################################################
modelList <- list ()
modelNms <- NULL
actualList <- list ()
predList <- list ()
modelTypeNms <- NULL


k <- 1
# model = RandomForest, type = all data, no undersample

modelNum <- k
################################################################################
#############Random Forest - multinomial ######################################
###################################################################################

#rfNms <- varSel_auc_0.007
#registerDoMC (7)
rfNms <- varSel_auc_0.007
myRF <- 
  randomForest (x=digitaltrain[,rfNms],
                y = as.factor (digitaltrain$target), 
                sampsize = c(1000, 100, 8, 10, 50, 200), ntree = 400, 
                nodesize = 1, do.trace = 50)
predictionMtx <- predict (myRF, digitalvalid[,rfNms], type = "prob")
save (myRF, predictionMtx, file = 'modelrfmultinom.rData')
bigactual <- digitalvalid$target

aucArray <- NULL
for (target in 0:5) {
  print (paste ("For Target:", target))
  actual <- ifelse (bigactual == target, 1, 0)
  myPred <- predictionMtx [,target+1]
  print (paste ("auc:", calc_auc(actual, myPred))) #0.7906861
  aucArray <- c(aucArray, calc_auc(actual, myPred))
  print (paste ("lift in top 2 percentile and top decile:", lift_model(actual,myPred,50))) # 11x in top percentile
  
}


modelList [[modelNum]] <- myRF
modelNms <- c(modelNms, "randomForestUnderSample")
actualList[[modelNum-1]] <- actual
predList[[modelNum]] <- predictionMtx
modelTypeNms <- c(modelTypeNms, "full")
save.image ('modeledRF.rData')
#8.3x in top 2 percentile
################################################################################
#############LIBLINEAR - multinomial ######################################
###################################################################################
library ('LiblineaR')

digitaltrain_t <- as.data.table (digitaltrain)
digitalvalid_t <- as.data.table (digitalvalid)
# normalize the matrices for regression
for (myNm in rfNms) {
  maxval <- max (digitaltrain_t[,myNm,with=F][[1]])
  digitaltrain_t [,myNm:=get (myNm)/ maxval, with=F]
  digitalvalid_t [,myNm:=get(myNm)/maxval, with=F]
}

trainMtx <- as.matrix (as.data.frame (digitaltrain_t[,rfNms,with=F]))
validMtx <- as.matrix (as.data.frame (digitalvalid_t[,rfNms,with=F]))

library (LiblineaR)
bigactual <- digitalvalid$target
#trainMtx <- as (trainMtx, "sparseMatrix")
#validMtx <- as (trainMtx, "sparseMatrix")

# 0 is L2 regularized; 6/7 is L1 regularized; 
#1 and 2 are L2 regularized with L2 Loss and are the same;
# 5 is L1 regularized, L2 loss

costArray <- c(1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1,1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
# is there are a more optimal cost

bestAUC <- 0
for (myCost in costArray) {
  myLiblinear <- LiblineaR (data=  trainMtx, target=as.factor (digitaltrain$target), 
                            type = 0, cost=myCost )
  predictionMtx <- predict (myLiblinear,  validMtx, proba=TRUE)
  predictionMtx <- predictionMtx[[2]]
  predictionMtx <- predictionMtx [,sapply (0:5, as.character)]
  aucArray <- NULL
  for (target in 1:5) {
    print (paste ("For Target:", target))
    actual <- ifelse (bigactual == target, 1, 0)
    myPred <- predictionMtx [,target+1]
    myAUC <- calc_auc(actual, myPred)
    print (paste ("auc:", myAUC )) #0.7906861
    print (paste ("lift in top 2 percentile and top decile:", lift_model(actual,myPred,50))) # 11x in top percentile
    aucArray <- c(aucArray, myAUC)
  }
  myAUC <- mean (aucArray)
  if (myAUC > bestAUC) {
    bestAUC <- myAUC
    bestCost <- myCost
    bestLiblinear <- myLiblinear
    bestPredictionMtx <- predictionMtx
  }	
  print (paste ('myCost', myCost, 'myAUC', myAUC ))
}


modelNum <- modelNum + 1
modelList [[modelNum]] <- bestLiblinear
modelNms <- c(modelNms, "liblinearfull")
actualList[[modelNum]] <- bigactual
predList[[modelNum]] <- bestPredictionMtx
modelTypeNms <- c(modelTypeNms, "full")
save.image ('modeledLiblinear.rData')



bestAUC <- 0
typeArray <- c(0, 6)
posweight <- sum (digitaltrain$target == 1)
negweight <- sum (digitaltrain$target == 0)


costArray <- c( 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1,1e2, 1e3, 1e4)
# is there are a more optimal cost

#
bestAUC <- 0
myWeights <- (1-as.numeric(table (digitaltrain$target)/nrow(digitaltrain)))
names (myWeights) <- names (table (digitaltrain$target))
for (myType in typeArray) {
  for (myCost in costArray) {
    myLiblinear <- LiblineaR (data=  trainMtx, target=as.factor (digitaltrain$target), 
                              type = myType, cost=myCost, wi= myWeights)
    predictionMtx <- predict (myLiblinear,  validMtx, proba=TRUE)
    predictionMtx <- predictionMtx[[2]]
    predictionMtx <- predictionMtx [,sapply (0:5, as.character)]
    aucArray <- NULL
    for (target in 1:5) {
      print (paste ("For Target:", target))
      actual <- ifelse (bigactual == target, 1, 0)
      myPred <- predictionMtx [,target+1]
      myAUC <- calc_auc(actual, myPred)
      #print (paste ("myCost", myCost, "myType", myType, "auc:", myAUC )) #0.7906861
      #print (paste ("lift in top 2 percentile and top decile:", lift_model(actual,myPred,50))) # 11x in top percentile
      aucArray <- c(aucArray, myAUC)
    }
    myAUC <- mean (aucArray)
    if (myAUC > bestAUC) {
      bestAUC <- myAUC
      bestCost <- myCost
      bestType <- myType
      bestLiblinear <- myLiblinear
      bestPredictionMtx <- predictionMtx
    }	
    print (paste ('myCost', myCost, 'myAUC', myAUC, 'myType', myType ))
  }
}



modelNum <- modelNum + 1
modelList [[modelNum]] <- bestLiblinear
modelNms <- c(modelNms, "liblinearfull")
actualList[[modelNum]] <- bigactual
predList[[modelNum]] <- bestPredictionMtx
modelTypeNms <- c(modelTypeNms, "full")

save (bestLiblinear, bestPredictionMtx, file = 'bestlinear.rData')
rm (trainMtx, validMtx, myNZV )
rm (digitaltrain_t, digitalvalid_t)
save.image ('modeledLiblinear_v2.rData')

################################################################################
#############XGBOOST - multinomial ######################################
###################################################################################

custom_auc_nonxgb <- function (bigactual, predictionMtx) {
  aucArray <- NULL
  for (target in 1:5) {		
    actual <- ifelse (bigactual == target, 1, 0)
    myPred <- predictionMtx [,target+1]
    myAUC <- calc_auc(actual, myPred)
    aucArray <- c(aucArray, myAUC)
  }
  print (aucArray)
  myAUC <- mean (aucArray)
  return (myAUC)
}

library ('xgboost')
xgbNms <-  varSel_auc_0.007
dtrain <- xgb.DMatrix (data=as.matrix(digitaltrain[,xgbNms]), label = digitaltrain$target)
dvalid <- xgb.DMatrix (data=as.matrix(digitalvalid[,xgbNms]), label = digitalvalid$target)

xgb.DMatrix.save (dtrain, 'xgb.dtrain')
xgb.DMatrix.save (dvalid, 'xgb.dvalid')
watchlist <- list (eval = dvalid, train= dtrain)

posweight <- sum (digitaltrain$target == 1)
negweight <- sum (digitaltrain$target == 0)

# 0.8866541 0.8598960 0.8362996 0.8521856 0.7823382 param <- list (max.depth = 5, eta = 0.7, silent = 1, nthread = 11, objective = "multi:softprob", eval_metric="auc",aximize = TRUE)
param <- list (max.depth = 9, eta = 0.2, silent = 0, nthread = 14, 
               objective = "multi:softprob", eval_metric="auc",maximize = TRUE)
bigxgboost <- xgb.train (params = param, data = dtrain, nrounds=205, watchlist=watchlist, 
                         verbose = 1, scale_pos_weight = negweight/posweight, num_class=6 )
predictionMtx <- predict (bigxgboost, dvalid)
predictionMtx <- matrix (data = predictionMtx, ncol = 6, byrow=TRUE)

custom_auc_nonxgb (bigactual, predictionMtx)

modelNum <- modelNum + 1
modelList [[modelNum]] <- bigxgboost
modelNms <- c(modelNms, "xgboostfull")
actualList[[modelNum]] <- bigactual
predList[[modelNum]] <- predictionMtx 
modelTypeNms <- c(modelTypeNms, "full")
xgb.save (bigxgboost, 'xgboost_auc.rData')



custom_auc_xgb <- function (preds, dtrainnew) {
  aucArray <- NULL
  bigactual <- getinfo (dtrainnew, "label")
  predictionMtx <- matrix (data = preds, ncol = 6, byrow=TRUE)
  for (target in 1:5) {		
    actual <- ifelse (bigactual == target, 1, 0)
    myPred <- predictionMtx [,target+1]
    myAUC <- calc_auc(actual, myPred)
    aucArray <- c(aucArray, myAUC)
  }
  myAUC <- mean (aucArray)
  return (list (metric = "custom_auc", value = myAUC))
}

# 0.8866541 0.8598960 0.8362996 0.8521856 0.7823382 param <- list (max.depth = 5, eta = 0.7, silent = 1, nthread = 11, objective = "multi:softprob", eval_metric="auc",aximize = TRUE)

param <- list (max.depth = 7, eta = 0.5, silent = 0, nthread = 15, objective = "multi:softprob", eval_metric="auc",
               maximize = TRUE, eval_metric="auc", feval=custom_auc_xgb)

aucxgboost <- xgb.train (params = param, data = dtrain, nrounds=15, watchlist=watchlist, 
                         scale_pos_weight = negweight/posweight, num_class=6 )
predictionMtx <- predict (aucxgboost, dvalid)
predictionMtx <- matrix (data = predictionMtx, ncol = 6, byrow=TRUE)

custom_auc_nonxgb (bigactual, predictionMtx)

modelNum <- modelNum + 1
modelList [[modelNum]] <- aucxgboost
modelNms <- c(modelNms, "xgboostfull")
actualList[[modelNum]] <- bigactual
predList[[modelNum]] <- predictionMtx 
modelTypeNms <- c(modelTypeNms, "full")
xgb.save (aucxgboost, 'xgboost_auc_short.rData')

multilogloss <- function (actual, predictionMtx) {
  # create actualMtx
  label <- as.factor (actual)
  fMtx <- as (label, "sparseMatrix")
  fMtx <- as.matrix (t (fMtx))
  eps <- 1e-5
  predictionMtx <- pmax (pmin (log1p (predictionMtx), 1-eps), eps)
  
  
  multilogloss <- (-1.0/length(actual)) * sum (fMtx * predictionMtx)
  return (multilogloss)
}



save.image ('xgboosted.rData')

modelNms <- c ("randomforestundersample", "liblinear", "liblinearweighted", "xgboost","smallxgboost")

for (i in 1:length(modelNms)) {
  print (i)
  colnames (predList [[i]]) <- paste (modelNms[i], 0:5, sep="_")
  temp <- as.data.frame (predList[[i]])
  setDT (temp)
  if (i == 1) stackDF <- temp
  if (i != 1) stackDF <- cbind (stackDF, temp)
  rm (temp)
}
setDT (digitalvalid)
digitalvalid [,mankid := 1:nrow(digitalvalid)]
stackDF [,email_id := digitalvalid$email]
stackDF [,mankid := digitalvalid$mankid]

fn_sample_stratified <- function (datatrain, numfolds, varstratify, numseed=11) {
  set.seed (numseed)
  num_levels <- length (unique (datatrain[,varstratify]))
  for (m in 0:num_levels) {              # generate a stratified sample
    indices <- rep ( c(1:numfolds), each = nrow(datatrain[datatrain[,varstratify]==m,])/numfolds)
    indices <- c(indices, rep ( 1, each = nrow(datatrain[datatrain[,varstratify]==m,])-length (indices)))
    datatrain[datatrain[,varstratify]==m,"indices"] <- sample (indices, length(indices))
  }
  return (datatrain)
}

stackDF [,target:=bigactual]
stackDF <- as.data.frame (stackDF)

stackDF <- fn_sample_stratified (stackDF, 2, "target")

label <- as.factor (stackDF$target)
fMtx <- as (label, "sparseMatrix")
fMtx <- as.matrix (t (fMtx))
colnames (fMtx) <- paste ("target", 0:5, sep="_")
setDT (stackDF)
stackDF <- cbind (stackDF, as.data.table (as.data.frame (fMtx)))
stackDF <- as.data.frame(stackDF)

library(nnls)

predDF <- NULL
for (i in c(1:2))    {
  temptrain <- stackDF[stackDF$indices != i,]
  tempvalid <- stackDF[stackDF$indices == i,]
  for (target in c(0:5)) {
    myTgt <- paste ("target", target, sep="_")
    myRHS <- paste (modelNms, target, sep="_")
    myForm <- paste (myTgt, paste (myRHS, collapse="+"), sep="~")
    
    myGLM <- nnls (as.matrix (temptrain[,myRHS]), temptrain[,myTgt] )
    myPred <-  as.matrix (tempvalid [,myRHS])%*%coef (myGLM)
    
    print (paste ("Predicted", target, calc_auc (tempvalid[,myTgt], myPred)))
    for (myNm in myRHS)
      print (paste (myNm, calc_auc (tempvalid[,myTgt], tempvalid[,myNm])))
    tempvalid [,paste("stack", target, sep="_")] <- myPred
  }
  predDF1 <- rbind (predDF1, tempvalid)
}

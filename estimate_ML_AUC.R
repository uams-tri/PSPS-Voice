###########################################################
# R code to estimate AUC for machine learning classifiers #
###########################################################

# Copyright (C) 2026 University of Arkansas for Medical Sciences
# Author: Yasir Rahmatallah, yrahmatallah@uams.edu
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Code was tested using R version 4.4.2, and the following package versions:
# randomForest_4.7-1.2, e1071_1.7-16, glmnet_4.1-10, caret_7.0-1, dplyr_1.1.4,  pROC_1.18.5, ROCR_1.0-11


library(MASS)
library(dplyr)
library(caret)
library(randomForest)
library(pROC)
library(ROCR)
library(e1071)
library(glmnet)

which.vowel <- "a" # one of "a", "i", or "u"
itr <- 100
kernel.type <- "radial"

feature.table <- read.csv(paste("path to file", "\\combined_features_", which.vowel, ".csv", sep=""))
combined.parselmouth.LPC <- as.matrix(feature.table[, c(3:36)])
combined.parselmouth.LAR <- as.matrix(feature.table[, c(c(3:16), c(37:56))])
combined.parselmouth.lpcep <- as.matrix(feature.table[, c(c(3:16), c(57:76))])
combined.parselmouth.lpmfcc <- as.matrix(feature.table[, c(c(3:16), c(77:96))])
rownames(combined.parselmouth.LPC) <- feature.table[, 1]
labels <- feature.table[, 2]

auc.te.RF <- auc.oob.RF <- auc.te.LR <- auc.tr.LR <- auc.te.SVM <- auc.tr.SVM <- matrix(0, 4, itr)
imp.RF.LPC <- imp.RF.LAR <- imp.RF.lpcep <- imp.RF.lpmfcc <- matrix(0, itr, ncol(combined.parselmouth.LPC))
colnames(imp.RF.LPC) <- colnames(imp.RF.LAR) <- colnames(imp.RF.lpcep) <- colnames(imp.RF.lpmfcc) <- colnames(combined.parselmouth.LPC)
pr.tr <- 2/3
pr.te <- 1/3

# start a loop to estimate AUC itr times
for(k in 1:itr){

#######################
# Random Forest - LPC #
#######################
x <- as_tibble(combined.parselmouth.LPC)
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
rf1 <- randomForest(data = x_tr, PP ~ ., ntree = 500,
type = "classification", mtry = 6, nodesize = 5, importance = TRUE)
rf1_te <- predict(rf1, newdata = x_te, type = "prob")
proc_te <- roc(response = x_te$PP, predictor = rf1_te[,2])
proc_oob <- roc(response = x_tr$PP, predictor = rf1$votes[,2])
auc.te.RF[1,k] <- auc(proc_te)
auc.oob.RF[1,k] <- auc(proc_oob)
imp.RF.LPC[k,] <- rf1$importance[,4] # MeanDecreaseGini

#######################
# Random Forest - LAR #
#######################
x <- as_tibble(combined.parselmouth.LAR) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
rf1 <- randomForest(data = x_tr, PP ~ ., ntree = 500,
type = "classification", mtry = 6, nodesize = 5, importance = TRUE)
rf1_te <- predict(rf1, newdata = x_te, type = "prob")
proc_te <- roc(response = x_te$PP, predictor = rf1_te[,2])
proc_oob <- roc(response = x_tr$PP, predictor = rf1$votes[,2])
auc.te.RF[2,k] <- auc(proc_te)
auc.oob.RF[2,k] <- auc(proc_oob)
imp.RF.LAR[k,] <- rf1$importance[,4] # MeanDecreaseGini

#########################
# Random Forest - lpcep #
#########################
x <- as_tibble(combined.parselmouth.lpcep) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
rf1 <- randomForest(data = x_tr, PP ~ ., ntree = 500,
type = "classification", mtry = 6, nodesize = 5, importance = TRUE)
rf1_te <- predict(rf1, newdata = x_te, type = "prob")
proc_te <- roc(response = x_te$PP, predictor = rf1_te[,2])
proc_oob <- roc(response = x_tr$PP, predictor = rf1$votes[,2])
auc.te.RF[3,k] <- auc(proc_te)
auc.oob.RF[3,k] <- auc(proc_oob)
imp.RF.lpcep[k,] <- rf1$importance[,4] # MeanDecreaseGini

##########################
# Random Forest - lpmfcc #
##########################
x <- as_tibble(combined.parselmouth.lpmfcc)
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
rf1 <- randomForest(data = x_tr, PP ~ ., ntree = 500,
type = "classification", mtry = 6, nodesize = 5, importance = TRUE)
rf1_te <- predict(rf1, newdata = x_te, type = "prob")
proc_te <- roc(response = x_te$PP, predictor = rf1_te[,2])
proc_oob <- roc(response = x_tr$PP, predictor = rf1$votes[,2])
auc.te.RF[4,k] <- auc(proc_te)
auc.oob.RF[4,k] <- auc(proc_oob)
imp.RF.lpmfcc[k,] <- rf1$importance[,4] # MeanDecreaseGini

###################################
# Ridge Logistic Regression - LPC # 
###################################
x <- combined.parselmouth.LPC
PP <- labels-1
train_index <- createDataPartition(PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
PP_tr <- PP[train_index]
PP_te <- PP[-train_index]
lambda_seq <- 10^seq(2, -2, by=-0.1)
cv.model <- cv.glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, lambda=lambda_seq, family="binomial", nfolds=5) 
ridge.model <- glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, family="binomial", lambda=cv.model$lambda.min)
pred_te <- predict(ridge.model, newx = as.matrix(x_te))
pred_tr <- predict(ridge.model, newx = as.matrix(x_tr))
rocr_tr <- prediction(predictions = pred_tr, labels = PP_tr)
rocr_te <- prediction(predictions = pred_te, labels = PP_te)
roc_tr <- performance(rocr_tr, measure = "tpr", x.measure = "fpr")
roc_te <- performance(rocr_te,  measure = "tpr", x.measure = "fpr")
auc_tr <- performance(rocr_tr, measure = "auc")@y.values
auc_te <- performance(rocr_te, measure = "auc")@y.values
auc.te.LR[1,k] <- unlist(auc_te)
auc.tr.LR[1,k] <- unlist(auc_tr)

###################################
# Ridge Logistic Regression - LAR # 
###################################
x <- combined.parselmouth.LAR
PP <- labels-1
train_index <- createDataPartition(PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
PP_tr <- PP[train_index]
PP_te <- PP[-train_index]
lambda_seq <- 10^seq(2, -2, by=-0.1)
cv.model <- cv.glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, lambda=lambda_seq, family="binomial", nfolds=5) 
ridge.model <- glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, family="binomial", lambda=cv.model$lambda.min)
pred_te <- predict(ridge.model, newx = as.matrix(x_te))
pred_tr <- predict(ridge.model, newx = as.matrix(x_tr))
rocr_tr <- prediction(predictions = pred_tr, labels = PP_tr)
rocr_te <- prediction(predictions = pred_te, labels = PP_te)
roc_tr <- performance(rocr_tr, measure = "tpr", x.measure = "fpr")
roc_te <- performance(rocr_te,  measure = "tpr", x.measure = "fpr")
auc_tr <- performance(rocr_tr, measure = "auc")@y.values
auc_te <- performance(rocr_te, measure = "auc")@y.values
auc.te.LR[2,k] <- unlist(auc_te)
auc.tr.LR[2,k] <- unlist(auc_tr)

#####################################
# Ridge Logistic Regression - lpcep # 
#####################################
x <- combined.parselmouth.lpcep
PP <- labels-1
train_index <- createDataPartition(PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
PP_tr <- PP[train_index]
PP_te <- PP[-train_index]
lambda_seq <- 10^seq(2, -2, by=-0.1)
cv.model <- cv.glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, lambda=lambda_seq, family="binomial", nfolds=5) 
ridge.model <- glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, family="binomial", lambda=cv.model$lambda.min)
pred_te <- predict(ridge.model, newx = as.matrix(x_te))
pred_tr <- predict(ridge.model, newx = as.matrix(x_tr))
rocr_tr <- prediction(predictions = pred_tr, labels = PP_tr)
rocr_te <- prediction(predictions = pred_te, labels = PP_te)
roc_tr <- performance(rocr_tr, measure = "tpr", x.measure = "fpr")
roc_te <- performance(rocr_te,  measure = "tpr", x.measure = "fpr")
auc_tr <- performance(rocr_tr, measure = "auc")@y.values
auc_te <- performance(rocr_te, measure = "auc")@y.values
auc.te.LR[3,k] <- unlist(auc_te)
auc.tr.LR[3,k] <- unlist(auc_tr)

######################################
# Ridge Logistic Regression - lpmfcc # 
######################################
x <- combined.parselmouth.lpmfcc
PP <- labels-1
train_index <- createDataPartition(PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
PP_tr <- PP[train_index]
PP_te <- PP[-train_index]
lambda_seq <- 10^seq(2, -2, by=-0.1)
cv.model <- cv.glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, lambda=lambda_seq, family="binomial", nfolds=5) 
ridge.model <- glmnet(x=as(x_tr, "sparseMatrix"), y=factor(PP_tr), alpha=0, family="binomial", lambda=cv.model$lambda.min)
pred_te <- predict(ridge.model, newx = as.matrix(x_te))
pred_tr <- predict(ridge.model, newx = as.matrix(x_tr))
rocr_tr <- prediction(predictions = pred_tr, labels = PP_tr)
rocr_te <- prediction(predictions = pred_te, labels = PP_te)
roc_tr <- performance(rocr_tr, measure = "tpr", x.measure = "fpr")
roc_te <- performance(rocr_te,  measure = "tpr", x.measure = "fpr")
auc_tr <- performance(rocr_tr, measure = "auc")@y.values
auc_te <- performance(rocr_te, measure = "auc")@y.values
auc.te.LR[4,k] <- unlist(auc_te)
auc.tr.LR[4,k] <- unlist(auc_tr)

################################
# Support Vector Machine - LPC #
################################
x <- as_tibble(combined.parselmouth.LPC) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
model <- svm(PP~., data=x_tr, scale=TRUE, kernel=kernel.type, probability=TRUE, cost=1)
pred.tr <- predict(model, x_tr, probability=TRUE)
pred.te <- predict(model, x_te, probability=TRUE)
prop.tr <- attr(pred.tr, "probabilities")
prop.te <- attr(pred.te, "probabilities")
proc_tr <- roc(response = x_tr$PP, predictor = prop.tr[,2])
proc_te <- roc(response = x_te$PP, predictor = prop.te[,2])
auc.tr.SVM[1,k] <- auc(proc_tr)
auc.te.SVM[1,k] <- auc(proc_te)

################################
# Support Vector Machine - LAR #
################################
x <- as_tibble(combined.parselmouth.LAR) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
model <- svm(PP~., data=x_tr, scale=TRUE, kernel=kernel.type, probability=TRUE, cost=1)
pred.tr <- predict(model, x_tr, probability=TRUE)
pred.te <- predict(model, x_te, probability=TRUE)
prop.tr <- attr(pred.tr, "probabilities")
prop.te <- attr(pred.te, "probabilities")
proc_tr <- roc(response = x_tr$PP, predictor = prop.tr[,2])
proc_te <- roc(response = x_te$PP, predictor = prop.te[,2])
auc.tr.SVM[2,k] <- auc(proc_tr)
auc.te.SVM[2,k] <- auc(proc_te)

##################################
# Support Vector Machine - lpcep #
##################################
x <- as_tibble(combined.parselmouth.lpcep) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
model <- svm(PP~., data=x_tr, scale=TRUE, kernel=kernel.type, probability=TRUE, cost=1)
pred.tr <- predict(model, x_tr, probability=TRUE)
pred.te <- predict(model, x_te, probability=TRUE)
prop.tr <- attr(pred.tr, "probabilities")
prop.te <- attr(pred.te, "probabilities")
proc_tr <- roc(response = x_tr$PP, predictor = prop.tr[,2])
proc_te <- roc(response = x_te$PP, predictor = prop.te[,2])
auc.tr.SVM[3,k] <- auc(proc_tr)
auc.te.SVM[3,k] <- auc(proc_te)

###################################
# Support Vector Machine - lpmfcc #
###################################
x <- as_tibble(combined.parselmouth.lpmfcc) 
x <- mutate(x, PP = ifelse(labels == 1, "yes", "no"))
x$PP <- factor(x$PP) 
train_index <- createDataPartition(x$PP, p=pr.tr, list=F, times=1)
x_tr <- x[train_index,]
x_te <- x[-train_index,]
model <- svm(PP~., data=x_tr, scale=TRUE, kernel=kernel.type, probability=TRUE, cost=1)
pred.tr <- predict(model, x_tr, probability=TRUE)
pred.te <- predict(model, x_te, probability=TRUE)
prop.tr <- attr(pred.tr, "probabilities")
prop.te <- attr(pred.te, "probabilities")
proc_tr <- roc(response = x_tr$PP, predictor = prop.tr[,2])
proc_te <- roc(response = x_te$PP, predictor = prop.te[,2])
auc.tr.SVM[4,k] <- auc(proc_tr)
auc.te.SVM[4,k] <- auc(proc_te)

} # end of for loop


save(auc.te.RF, auc.oob.RF, auc.te.LR, auc.tr.LR, auc.te.SVM, auc.tr.SVM,
imp.RF.LPC, imp.RF.LAR, imp.RF.lpcep, imp.RF.lpmfcc, file="FileName.RData") 

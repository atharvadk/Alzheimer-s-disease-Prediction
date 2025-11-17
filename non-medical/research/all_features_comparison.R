#install.packages(c("caret", "randomForest", "xgboost", "e1071", "nnaivebayes", "adabag", "rpart", "ROCR", "mlbench"))

library(caret)
library(randomForest)
library(xgboost) #xgboost
library(e1071)  # SVM
library(nnet)   # Neural Network
library(naivebayes)
library(adabag) # AdaBoost
library(rpart)  # Decision Tree
library(ROCR)   # ROC-AUC Calculation


Sys.setenv(XGBOOST_VERBOSITY = 0)

data <- read.csv("rfe_columns.csv")

data$Diagnosis <- as.factor(data$Diagnosis)

levels(data$Diagnosis) <- make.names(levels(data$Diagnosis))

data <- na.omit(data)

set.seed(42)
trainIndex <- createDataPartition(data$Diagnosis, p=0.7, list=FALSE)
trainData <- data[trainIndex, c(1:13, ncol(data))]  
testData  <- data[-trainIndex, c(1:13, ncol(data))]  


train_control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)

models <- list(
  "Logistic Regression" = train(Diagnosis ~ ., data=trainData, method="glm", family="binomial", trControl=train_control),
  "Random Forest" = train(Diagnosis ~ ., data=trainData, method="rf", trControl=train_control),
  "Gradient Boosting" = train(Diagnosis ~ ., data=trainData, method="xgbTree", trControl=train_control),
  "SVM" = train(Diagnosis ~ ., data=trainData, method="svmRadial", trControl=train_control),
  "KNN" = train(Diagnosis ~ ., data=trainData, method="knn", trControl=train_control),
  "Naive Bayes" = train(Diagnosis ~ ., data=trainData, method="naive_bayes", trControl=train_control),
  "Decision Tree" = train(Diagnosis ~ ., data=trainData, method="rpart", trControl=train_control),
  "Neural Network" = train(Diagnosis ~ ., data=trainData, method="nnet", trace=FALSE, trControl=train_control),
  "AdaBoost" = train(Diagnosis ~ ., data=trainData, method="AdaBoost.M1", trControl=train_control)
)

evaluate_model <- function(model, testData) {
  predictions <- predict(model, newdata=testData)
  if (any(is.na(predictions))) {
    return(c(Accuracy=NA, Precision=NA, Recall=NA, F1=NA, AUC=NA))
  }
  
  probabilities <- predict(model, newdata=testData, type="prob")[,2]
  
  confusion <- confusionMatrix(predictions, testData$Diagnosis)
  accuracy <- confusion$overall["Accuracy"]
  precision <- confusion$byClass["Pos Pred Value"]
  recall <- confusion$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  pred <- prediction(probabilities, testData$Diagnosis)
  auc <- performance(pred, "auc")@y.values[[1]]
  
  return(c(Accuracy=accuracy, Precision=precision, Recall=recall, F1=f1, AUC=auc))
}

results <- sapply(models, evaluate_model, testData=testData)

results_df <- as.data.frame(t(results))
results_df <- results_df[order(-results_df$Accuracy),]

print(results_df)
write.csv(results_df, "70_30_rfe_first13.csv", row.names=TRUE)

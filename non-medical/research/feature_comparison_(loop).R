
library(caret)
library(randomForest)
library(xgboost)
library(e1071)  
library(nnet)   
library(naivebayes)
library(adabag) 
library(rpart)  
library(ROCR) 
library(mlbench)

Sys.setenv(XGBOOST_VERBOSITY = 0)

data <- read.csv("rf_columns.csv")

data$Diagnosis <- as.factor(data$Diagnosis)

levels(data$Diagnosis) <- make.names(levels(data$Diagnosis))

data <- na.omit(data)

set.seed(42)

train_control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)

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

split_ratios <- c(0.7, 0.75, 0.8)

all_split_results <- list()

for (split_ratio in split_ratios) {
  cat(paste("\n\n========================================"))
  cat(paste("\nRunning models with train/test split:", split_ratio, "/", round(1-split_ratio, 2)))
  cat(paste("\n========================================\n"))
  
  trainIndex <- createDataPartition(data$Diagnosis, p=split_ratio, list=FALSE)
  
  all_results <- list()
  
  for (num_cols in 10:13) {
    cat(paste("\nEvaluating models with first", num_cols, "columns plus Diagnosis\n"))
    
    # Prepare data subsets with specific number of columns plus Diagnosis
    trainData <- data[trainIndex, c(1:num_cols, ncol(data))]
    testData <- data[-trainIndex, c(1:num_cols, ncol(data))]
    
    # Define models
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
    
    # Evaluate all models
    results <- sapply(models, evaluate_model, testData=testData)
    
    # Convert results to DataFrame and sort by accuracy
    results_df <- as.data.frame(t(results))
    results_df <- results_df[order(-results_df$Accuracy),]
    
    # Store results
    all_results[[paste0("cols_", num_cols)]] <- results_df
    
    # Print results
    cat("\nResults for first", num_cols, "columns:\n")
    print(results_df)
    
    filename <- paste0(round(split_ratio*100), "_", round((1-split_ratio)*100), "_rf_first", num_cols)
    write.csv(results_df, paste0(filename, ".csv"), row.names=TRUE)
  }
  
  summary_df <- data.frame(
    NumColumns = 10:13,
    BestModel = sapply(all_results, function(df) rownames(df)[1]),
    BestAccuracy = sapply(all_results, function(df) df$Accuracy[1]),
    BestAUC = sapply(all_results, function(df) df$AUC[1])
  )
  
  cat("\nSummary of best models across different column sets with", 
      split_ratio*100, "/", (1-split_ratio)*100, "split:\n")
  print(summary_df)
  
  write.csv(summary_df, paste0("summary_column_comparison_summary_", 
                               round(split_ratio*100), "_", 
                               round((1-split_ratio)*100), ".csv"), 
            row.names=FALSE)
  
  all_split_results[[paste0("split_", split_ratio)]] <- summary_df
}

final_comparison <- data.frame(
  SplitRatio = rep(split_ratios, each=4),
  NumColumns = rep(10:13, length(split_ratios)),
  BestModel = unlist(lapply(all_split_results, function(df) df$BestModel)),
  BestAccuracy = unlist(lapply(all_split_results, function(df) df$BestAccuracy)),
  BestAUC = unlist(lapply(all_split_results, function(df) df$BestAUC))
)

cat("\n\n========================================")
cat("\nFinal comparison across all splits and column sets:")
cat("\n========================================\n")
print(final_comparison)

write.csv(final_comparison, "rf_final_comparison_all_splits.csv", row.names=FALSE)
# Install & Load required libraries

library(caret)
library(xgboost)
library(randomForest)
library(glmnet)
library(e1071)  # For F1-score

# Load the datasets
rfe_data <- read.csv("Top12_RFE_Features.csv")
lasso_data <- read.csv("Top12_Lasso_Features.csv")
rf_data <- read.csv("Top12_RF_Features.csv")

# Function to train and evaluate models
train_and_evaluate <- function(data, dataset_name) {
  
  # Convert target variable to factor
  data$Diagnosis <- as.factor(data$Diagnosis)
  
  # Split into training & testing (80-20 split)
  set.seed(42)
  trainIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  #### 🔹 1️⃣ Train Logistic Regression
  logistic_model <- glm(Diagnosis ~ ., data=trainData, family="binomial")
  logistic_preds <- predict(logistic_model, testData, type="response")
  logistic_preds <- ifelse(logistic_preds > 0.5, 1, 0)
  
  #### 🔹 2️⃣ Train Random Forest
  rf_model <- randomForest(Diagnosis ~ ., data=trainData, ntree=200)
  rf_preds <- predict(rf_model, testData)
  
  #### 🔹 3️⃣ Train XGBoost
  X_train <- as.matrix(trainData[, !names(trainData) %in% "Diagnosis"])
  X_test <- as.matrix(testData[, !names(testData) %in% "Diagnosis"])
  y_train <- as.numeric(trainData$Diagnosis) - 1
  y_test <- as.numeric(testData$Diagnosis) - 1
  
  xgb_model <- xgboost(data=X_train, label=y_train, nrounds=100, objective="binary:logistic")
  xgb_preds <- predict(xgb_model, X_test)
  xgb_preds <- ifelse(xgb_preds > 0.5, 1, 0)
  
  # Convert predictions to factor
  logistic_preds <- as.factor(logistic_preds)
  rf_preds <- as.factor(rf_preds)
  xgb_preds <- as.factor(xgb_preds)
  testData$Diagnosis <- as.factor(y_test)
  
  # Evaluation metrics
  logistic_cm <- confusionMatrix(logistic_preds, testData$Diagnosis)
  rf_cm <- confusionMatrix(rf_preds, testData$Diagnosis)
  xgb_cm <- confusionMatrix(xgb_preds, testData$Diagnosis)
  
  # Store results
  results <- data.frame(
    Dataset = dataset_name,
    Model = c("Logistic Regression", "Random Forest", "XGBoost"),
    Accuracy = c(logistic_cm$overall["Accuracy"], rf_cm$overall["Accuracy"], xgb_cm$overall["Accuracy"]),
    Precision = c(logistic_cm$byClass["Precision"], rf_cm$byClass["Precision"], xgb_cm$byClass["Precision"]),
    Recall = c(logistic_cm$byClass["Recall"], rf_cm$byClass["Recall"], xgb_cm$byClass["Recall"]),
    F1_Score = c(logistic_cm$byClass["F1"], rf_cm$byClass["F1"], xgb_cm$byClass["F1"])
  )
  
  return(results)
}

# Train & evaluate on all three feature sets
results_rfe <- train_and_evaluate(rfe_data, "RFE")
results_lasso <- train_and_evaluate(lasso_data, "Lasso")
results_rf <- train_and_evaluate(rf_data, "Random Forest")

# Combine results
final_results <- rbind(results_rfe, results_lasso, results_rf)

# Save results
write.csv(final_results, "Model_Comparison_Results.csv", row.names=FALSE)

# Print results
print("✅ Model comparison saved in Model_Comparison_Results.csv")
print(final_results)

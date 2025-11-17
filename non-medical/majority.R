# Load required libraries
library(caret)
library(randomForest)
library(xgboost)
library(adabag) # AdaBoost
library(ROCR)   # ROC-AUC Calculation

# Suppress XGBoost warnings
Sys.setenv(XGBOOST_VERBOSITY = 0)

# Load dataset
data <- read.csv("rf_nonMedical_columns.csv")

# Ensure 'Diagnosis' is a factor (for classification)
data$Diagnosis <- as.factor(data$Diagnosis)

# Fix class names (convert them to valid R variable names)
levels(data$Diagnosis) <- make.names(levels(data$Diagnosis))

# Handle missing values by removing rows with NA
data <- na.omit(data)

# Split data into training (75%) and test (25%)
set.seed(42)
trainIndex <- createDataPartition(data$Diagnosis, p=0.75, list=FALSE)
trainData <- data[trainIndex, c(1:8, ncol(data))]  # Keep first 13 columns + Diagnosis
testData <- data[-trainIndex, c(1:8, ncol(data))]  # Keep first 13 columns + Diagnosis

# Define cross-validation control
train_control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)

# Define the tree-based models only
models <- list(
  "Random Forest" = train(Diagnosis ~ ., data=trainData, method="rf", trControl=train_control),
  "Gradient Boosting" = train(Diagnosis ~ ., data=trainData, method="xgbTree", trControl=train_control),
  "AdaBoost" = train(Diagnosis ~ ., data=trainData, method="AdaBoost.M1", trControl=train_control)
)

# Function to evaluate models
evaluate_model <- function(model, testData) {
  # Ensure no NAs in predictions
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
  
  # ROC-AUC calculation
  pred <- prediction(probabilities, testData$Diagnosis)
  auc <- performance(pred, "auc")@y.values[[1]]
  
  return(c(Accuracy=accuracy, Precision=precision, Recall=recall, F1=f1, AUC=auc))
}

# Create a dataframe to store predictions
prediction_df <- data.frame(ActualDiagnosis = testData$Diagnosis)

# Evaluate all models and store predictions
prediction_matrices <- list()  # To store raw predictions for majority voting

for (model_name in names(models)) {
  model <- models[[model_name]]
  
  # Get predictions (class labels)
  predictions <- predict(model, newdata=testData)
  
  # Get probabilities for positive class
  probabilities <- predict(model, newdata=testData, type="prob")[,2]
  
  # Add predictions to dataframe
  prediction_df[[paste0(model_name, "_Prediction")]] <- predictions
  prediction_df[[paste0(model_name, "_Probability")]] <- probabilities
  
  # Store raw predictions for majority voting
  prediction_matrices[[model_name]] <- as.numeric(predictions) - 1  # Convert to 0/1
}

# Create a matrix of predictions (each column is a model's predictions)
prediction_matrix <- do.call(cbind, prediction_matrices)

# Calculate majority vote (mode of each row)
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Apply majority voting row by row
majority_votes <- apply(prediction_matrix, 1, get_mode)

# Convert back to factor with the same levels as the original
majority_votes_factor <- factor(majority_votes, levels=c(0, 1), 
                                labels=levels(testData$Diagnosis))

# Add majority vote to prediction dataframe
prediction_df$Majority_Vote_Prediction <- majority_votes_factor

# Calculate ensemble performance metrics
ensemble_confusion <- confusionMatrix(prediction_df$Majority_Vote_Prediction, testData$Diagnosis)
ensemble_accuracy <- ensemble_confusion$overall["Accuracy"]
ensemble_precision <- ensemble_confusion$byClass["Pos Pred Value"]
ensemble_recall <- ensemble_confusion$byClass["Sensitivity"]
ensemble_f1 <- 2 * ((ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall))

# Calculate ensemble probabilities (average of model probabilities)
ensemble_probs <- rowMeans(data.frame(
  prediction_df$`Random Forest_Probability`,
  prediction_df$`Gradient Boosting_Probability`,
  prediction_df$`AdaBoost_Probability`
))
prediction_df$Majority_Vote_Probability <- ensemble_probs

# Calculate ensemble AUC
ensemble_pred <- prediction(ensemble_probs, testData$Diagnosis)
ensemble_auc <- performance(ensemble_pred, "auc")@y.values[[1]]

# Compute performance metrics for individual models
results <- sapply(models, evaluate_model, testData=testData)

# Convert results to DataFrame
results_df <- as.data.frame(t(results))

# Add ensemble results
ensemble_results <- c(Accuracy=ensemble_accuracy, Precision=ensemble_precision, 
                      Recall=ensemble_recall, F1=ensemble_f1, AUC=ensemble_auc)
results_df <- rbind(results_df, "Ensemble (Majority Vote)"=ensemble_results)

# Sort by accuracy
results_df <- results_df[order(-results_df$Accuracy),]

# Print results
print(results_df)

# Save performance metrics
write.csv(results_df, "75_25_rfe_treemodels_performance.csv", row.names=TRUE)

# Save predictions
write.csv(prediction_df, "75_25_rfe_treemodels_predictions.csv", row.names=FALSE)

# Print sample of predictions with majority vote
cat("\nSample of predictions with majority voting (first 10 rows):\n")
print(head(prediction_df, 10))

# Print confusion matrix for ensemble
cat("\nConfusion Matrix for Ensemble (Majority Vote):\n")
print(ensemble_confusion$table)
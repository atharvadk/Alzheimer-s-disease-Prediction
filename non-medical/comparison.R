# Load necessary libraries
library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(dplyr)

# Load the dataset
df <- read.csv("nonMedical_columns.csv")

# Separate features and target variable
X <- df %>% select(-Diagnosis)
y <- df$Diagnosis

# Standardize the features
X_scaled <- scale(X)

# Split dataset into training & testing sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[trainIndex, ]
X_test <- X_scaled[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# --- Feature Selection ---

# RFE
set.seed(42)
rfe_control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rfe_model <- rfe(X_train, y_train, sizes = 12, rfeControl = rfe_control)
rfe_features <- predictors(rfe_model)

# LASSO
set.seed(42)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")
lasso_coef <- coef(lasso_model, s = "lambda.min")
lasso_features <- rownames(lasso_coef)[lasso_coef[, 1] != 0][-1]  # Remove intercept

# Random Forest Feature Importance
set.seed(42)
rf_model <- randomForest(X_train, as.factor(y_train), ntree = 100, importance = TRUE)
rf_importance <- importance(rf_model)
rf_features <- names(sort(rf_importance[, 1], decreasing = TRUE))[1:12]

# --- Jaccard Similarity ---
jaccard_index <- function(set1, set2) {
  length(intersect(set1, set2)) / length(union(set1, set2))
}

jaccard_rfe_lasso <- jaccard_index(rfe_features, lasso_features)
jaccard_rfe_rf <- jaccard_index(rfe_features, rf_features)
jaccard_lasso_rf <- jaccard_index(lasso_features, rf_features)

cat("Jaccard Similarity:\n")
cat("RFE vs LASSO:", jaccard_rfe_lasso, "\n")
cat("RFE vs RF:", jaccard_rfe_rf, "\n")
cat("LASSO vs RF:", jaccard_lasso_rf, "\n")

# --- Model Training & Evaluation ---

evaluate_model <- function(X_selected, y, feature_set_name) {
  X_train_selected <- X_train[, X_selected]
  X_test_selected <- X_test[, X_selected]
  
  models <- list(
    "Logistic Regression" = train(X_train_selected, y_train, method = "glm", family = "binomial"),
    "Random Forest" = train(X_train_selected, as.factor(y_train), method = "rf"),
    "SVM" = train(X_train_selected, as.factor(y_train), method = "svmRadial")
  )
  
  results <- data.frame()
  
  for (model_name in names(models)) {
    model <- models[[model_name]]
    preds <- predict(model, X_test_selected)
    
    accuracy <- mean(preds == y_test)
    precision <- posPredValue(preds, y_test, positive = "1")
    recall <- sensitivity(preds, y_test, positive = "1")
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    # Calculate ROC-AUC if possible
    if ("prob" %in% names(model)) {
      prob_preds <- predict(model, X_test_selected, type = "prob")[,2]
      roc_auc <- roc(y_test, prob_preds)$auc
    } else {
      roc_auc <- NA
    }
    
    results <- rbind(results, data.frame(
      FeatureSet = feature_set_name,
      Model = model_name,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      F1_Score = f1,
      ROC_AUC = roc_auc
    ))
  }
  
  return(results)
}

# Run model evaluation for each feature selection method
final_results <- rbind(
  evaluate_model(rfe_features, y, "RFE"),
  evaluate_model(lasso_features, y, "LASSO"),
  evaluate_model(rf_features, y, "Random Forest")
)

# Save results to CSV
write.csv(final_results, "model_performance_comparison.csv", row.names = FALSE)

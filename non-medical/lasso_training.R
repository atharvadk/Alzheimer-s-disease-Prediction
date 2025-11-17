library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)
library(adabag)
library(klaR)

# Load dataset
df_lasso <- read.csv("lasso_selected_features.csv")

# Split features and target variable
X_lasso <- df_lasso[, !names(df_lasso) %in% c("Diagnosis")]
y_lasso <- as.factor(df_lasso$Diagnosis)

# Standardize features
preprocess_params_lasso <- preProcess(X_lasso, method = c("center", "scale"))
X_lasso_scaled <- predict(preprocess_params_lasso, X_lasso)

# Split into train and test sets
set.seed(42)
trainIndex_lasso <- createDataPartition(y_lasso, p = 0.8, list = FALSE)
X_lasso_train <- X_lasso_scaled[trainIndex_lasso, ]
y_lasso_train <- y_lasso[trainIndex_lasso]
X_lasso_test <- X_lasso_scaled[-trainIndex_lasso, ]
y_lasso_test <- y_lasso[-trainIndex_lasso]

# Define models
models_lasso <- list(
  "Logistic Regression" = train(X_lasso_train, y_lasso_train, method = "glm", family = "binomial"),
  "Decision Tree" = train(X_lasso_train, y_lasso_train, method = "rpart"),
  "Random Forest" = train(X_lasso_train, y_lasso_train, method = "rf"),
  "Gradient Boosting" = train(X_lasso_train, y_lasso_train, method = "gbm", verbose = FALSE),
  "Support Vector Machine" = train(X_lasso_train, y_lasso_train, method = "svmRadial"),
  "K-Nearest Neighbors" = train(X_lasso_train, y_lasso_train, method = "knn"),
  "Naive Bayes" = train(X_lasso_train, y_lasso_train, method = "nb"),
  "AdaBoost" = train(X_lasso_train, y_lasso_train, method = "adaboost"),
  "Extra Trees" = train(X_lasso_train, y_lasso_train, method = "treebag")
)

# Evaluate models
results_lasso <- data.frame(Model = character(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1_score = numeric())

for (name in names(models_lasso)) {
  model <- models_lasso[[name]]
  predictions <- predict(model, X_lasso_test)
  cm <- confusionMatrix(predictions, y_lasso_test)
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  results_lasso <- rbind(results_lasso, data.frame(Model = name, Accuracy = accuracy, Precision = precision, Recall = recall, F1_score = f1))
}

# Save results
write.csv(results_lasso, "classification_comparison_lasso_r.csv", row.names = FALSE)

print("Comparison table saved as classification_comparison_lasso_r.csv")

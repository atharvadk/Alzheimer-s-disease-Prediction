library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)
library(adabag)
library(klaR)
library(xgboost)

# Load dataset
df <- read.csv("rf_selected_features.csv")

# Split features and target variable
X <- df[, !names(df) %in% c("Diagnosis")]
y <- as.factor(df$Diagnosis)

# Standardize features
preprocess_params <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preprocess_params, X)

# Split into train and test sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X_scaled[-trainIndex, ]
y_test <- y[-trainIndex]

# Define models
models <- list(
  "Logistic Regression" = train(X_train, y_train, method = "glm", family = "binomial"),
  "Decision Tree" = train(X_train, y_train, method = "rpart"),
  "Random Forest" = train(X_train, y_train, method = "rf"),
  "Gradient Boosting" = train(X_train, y_train, method = "gbm", verbose = FALSE),
  "Support Vector Machine" = train(X_train, y_train, method = "svmRadial"),
  "K-Nearest Neighbors" = train(X_train, y_train, method = "knn"),
  "Naive Bayes" = train(X_train, y_train, method = "nb"),
  "AdaBoost" = train(X_train, y_train, method = "adaboost"),
  "Extra Trees" = train(X_train, y_train, method = "treebag"),
  "XGBoost" = train(X_train, y_train, method = "xgbTree")
)

# Evaluate models
results <- data.frame(Model = character(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1_score = numeric())

for (name in names(models)) {
  model <- models[[name]]
  predictions <- predict(model, X_test)
  cm <- confusionMatrix(predictions, y_test)
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  results <- rbind(results, data.frame(Model = name, Accuracy = accuracy, Precision = precision, Recall = recall, F1_score = f1))
}

# Save results
write.csv(results, "classification_comparison_r.csv", row.names = FALSE)

print("Comparison table saved as classification_comparison_r.csv")
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(nnet)
library(adabag)
library(klaR)

# Load dataset
df_rfe <- read.csv("rfe_selected_features.csv")

# Split features and target variable
X_rfe <- df_rfe[, !names(df_rfe) %in% c("Diagnosis")]
y_rfe <- as.factor(df_rfe$Diagnosis)

# Standardize features
preprocess_params_rfe <- preProcess(X_rfe, method = c("center", "scale"))
X_rfe_scaled <- predict(preprocess_params_rfe, X_rfe)

# Split into train and test sets
set.seed(42)
trainIndex_rfe <- createDataPartition(y_rfe, p = 0.8, list = FALSE)
X_rfe_train <- X_rfe_scaled[trainIndex_rfe, ]
y_rfe_train <- y_rfe[trainIndex_rfe]
X_rfe_test <- X_rfe_scaled[-trainIndex_rfe, ]
y_rfe_test <- y_rfe[-trainIndex_rfe]

# Define models
models_rfe <- list(
  "Logistic Regression" = train(X_rfe_train, y_rfe_train, method = "glm", family = "binomial"),
  "Decision Tree" = train(X_rfe_train, y_rfe_train, method = "rpart"),
  "Random Forest" = train(X_rfe_train, y_rfe_train, method = "rf", tuneGrid = data.frame(mtry = 5)),
  "Gradient Boosting" = train(X_rfe_train, y_rfe_train, method = "gbm", verbose = FALSE),
  "Support Vector Machine" = train(X_rfe_train, y_rfe_train, method = "svmRadial"),
  "K-Nearest Neighbors" = train(X_rfe_train, y_rfe_train, method = "knn"),
  "Naive Bayes" = train(X_rfe_train, y_rfe_train, method = "nb"),
  "AdaBoost" = train(X_rfe_train, y_rfe_train, method = "adaboost"),
  "Extra Trees" = train(X_rfe_train, y_rfe_train, method = "treebag")
)

# Evaluate models
results_rfe <- data.frame(Model = character(), Accuracy = numeric(), Precision = numeric(), Recall = numeric(), F1_score = numeric())

for (name in names(models_rfe)) {
  model <- models_rfe[[name]]
  predictions <- predict(model, X_rfe_test)
  cm <- confusionMatrix(predictions, y_rfe_test)
  
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  
  results_rfe <- rbind(results_rfe, data.frame(Model = name, Accuracy = accuracy, Precision = precision, Recall = recall, F1_score = f1))
}

# Save results
write.csv(results_rfe, "classification_comparison_rfe_r.csv", row.names = FALSE)

print("Comparison table saved as classification_comparison_rfe_r.csv")

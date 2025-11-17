
if (!require(caret)) install.packages("caret")
if (!require(randomForest)) install.packages("randomForest")


library(caret)
library(randomForest)


set.seed(123)


data <- read.csv("balanced_alzheimers_data.csv")

x <- data[, 1:32]  # Features
y <- data[, 33]    # Target: Species

control <- rfeControl(
  functions = rfFuncs,          # Use random forest as the base model
  method = "cv",                # Cross-validation
  number = 10,                  # 10-fold CV
  verbose = TRUE                # Print progress
)

rfe_results <- rfe(
  x = x,                        # Features
  y = y,                        # Target variable
  sizes = c(1:ncol(x)),         # Evaluate all possible feature subsets
  rfeControl = control          # Control parameters
)

print(rfe_results)

plot(rfe_results, type = c("g", "o"))

importance_scores <- varImp(rfe_results, scale = FALSE)
print(importance_scores)

plot(importance_scores)

optimal_features <- predictors(rfe_results)
cat("\nOptimal feature set:\n")
print(optimal_features)

feature_ranking <- rfe_results$variables
print(feature_ranking)
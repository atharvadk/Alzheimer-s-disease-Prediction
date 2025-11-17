# Load necessary libraries
library(caret)
library(glmnet)
library(randomForest)
library(dplyr)

# Load the dataset
df <- read.csv("nonMedical_columns.csv")

# Separate features and target variable
X <- df %>% select(-Diagnosis)
y <- df$Diagnosis

# Standardize the features
X_scaled <- scale(X)

# Split the dataset into training and test sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[trainIndex, ]
X_test <- X_scaled[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# RFE Feature Selection
set.seed(42)
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rfe_model <- rfe(X_train, y_train, sizes = 12, rfeControl = control)
rfe_features <- predictors(rfe_model)

# LASSO Feature Selection
set.seed(42)
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial")
lasso_coef <- coef(lasso_model, s = "lambda.min")
lasso_features <- rownames(lasso_coef)[lasso_coef[, 1] != 0][-1] # Remove intercept

# Random Forest Feature Selection
set.seed(42)
rf_model <- randomForest(X_train, as.factor(y_train), ntree = 100, importance = TRUE)
rf_importance <- importance(rf_model)
rf_features <- names(sort(rf_importance[, 1], decreasing = TRUE))[1:12]

# Create new datasets with selected features
rfe_df <- df %>% select(all_of(rfe_features), Diagnosis)
lasso_df <- df %>% select(all_of(lasso_features), Diagnosis)
rf_df <- df %>% select(all_of(rf_features), Diagnosis)

# Save datasets to CSV
write.csv(rfe_df, "rfe_selected_features.csv", row.names = FALSE)
write.csv(lasso_df, "lasso_selected_features.csv", row.names = FALSE)
write.csv(rf_df, "rf_selected_features.csv", row.names = FALSE)

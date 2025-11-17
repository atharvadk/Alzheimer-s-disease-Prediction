# Install and load necessary libraries
#install.packages(c("caret", "randomForest", "glmnet"))
library(caret)
library(randomForest)
library(glmnet)

# Load dataset
data <- read.csv("balanced_alzheimers_data.csv")

# Convert target variable to factor
data$Diagnosis <- as.factor(data$Diagnosis)

# Separate features and target variable
X <- data[, !names(data) %in% c("Diagnosis")]
y <- data$Diagnosis

### 🔹 1️⃣ RFE Feature Selection
set.seed(42)

# Define RFE control using Random Forest
control <- rfeControl(functions=rfFuncs, method="cv", number=5)

# Run RFE
rfe_model <- rfe(X, y, sizes=c(5, 10, 15, 20), rfeControl=control)

# Get top 12 features from RFE
rfe_selected_features <- head(predictors(rfe_model), 12)

# Save dataset with selected RFE features
rfe_data <- data[, c(rfe_selected_features, "Diagnosis")]
write.csv(rfe_data, "Top12_RFE_Features.csv", row.names=FALSE)

### 🔹 2️⃣ Lasso Regression Feature Selection
set.seed(42)

# Convert data to matrix format for glmnet
X_matrix <- as.matrix(X)

# Train Lasso model with cross-validation
lasso_model <- cv.glmnet(X_matrix, y, family="binomial", alpha=1)

# Get nonzero coefficients (excluding intercept)
lasso_coef <- coef(lasso_model, s="lambda.min")
lasso_importance_df <- data.frame(Feature = rownames(lasso_coef), Importance = abs(as.vector(lasso_coef)))
lasso_importance_df <- lasso_importance_df[lasso_importance_df$Feature != "(Intercept)", ]  # Remove intercept
lasso_importance_df <- lasso_importance_df[order(-lasso_importance_df$Importance), ]  # Sort by importance

# Get top 12 Lasso features
lasso_selected_features <- head(lasso_importance_df$Feature, 12)

# Save dataset with selected Lasso features
lasso_data <- data[, c(lasso_selected_features, "Diagnosis")]
write.csv(lasso_data, "Top12_Lasso_Features.csv", row.names=FALSE)

### 🔹 3️⃣ Random Forest Feature Selection
set.seed(42)

# Train Random Forest model
rf_model <- randomForest(X, y, ntree=100, importance=TRUE)

# Get feature importance
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(Feature=rownames(rf_importance), Importance=rf_importance[, "MeanDecreaseGini"])
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]  # Sort by importance

# Get top 12 Random Forest features
rf_selected_features <- head(rf_importance_df$Feature, 12)

# Save dataset with selected RF features
rf_data <- data[, c(rf_selected_features, "Diagnosis")]
write.csv(rf_data, "Top12_RF_Features.csv", row.names=FALSE)

print("✅ Top 12 features for RFE, Lasso, and Random Forest saved to CSV files!")

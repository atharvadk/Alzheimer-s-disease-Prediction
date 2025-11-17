# Install required packages if not already installed
#install.packages(c("caret", "randomForest", "glmnet", "ggplot2"))

# Load necessary libraries
library(caret)
library(randomForest)
library(glmnet)
library(ggplot2)

# Load dataset
data <- read.csv("balanced_alzheimers_data.csv")

# Convert target variable to factor
data$Diagnosis <- as.factor(data$Diagnosis)

# Separate features and target variable
X <- data[, !names(data) %in% c("Diagnosis")]
y <- data$Diagnosis

### 🔹 1️⃣ Recursive Feature Elimination (RFE) with Random Forest
set.seed(42)

# Define RFE control using Random Forest
control <- rfeControl(functions=rfFuncs, method="cv", number=5)

# Run Recursive Feature Elimination (RFE)
rfe_model <- rfe(X, y, sizes=c(5, 10, 15, 20), rfeControl=control)

# Get selected features from RFE
rfe_selected_features <- predictors(rfe_model)
print("✅ RFE Selected Features:")
print(rfe_selected_features)

# Get feature importance from the final trained model (Random Forest)
rf_importance <- importance(rfe_model$fit)

# Convert importance to a dataframe
rfe_importance_df <- data.frame(Feature=rownames(rf_importance), Importance=rf_importance[, "MeanDecreaseGini"])

# Sort by importance
rfe_importance_df <- rfe_importance_df[order(-rfe_importance_df$Importance), ]
print("✅ RFE Feature Importance Scores:")
print(rfe_importance_df)

### 🔹 Save Results to CSV
write.csv(rfe_importance_df, "RFE_Feature_Importance.csv", row.names=FALSE)

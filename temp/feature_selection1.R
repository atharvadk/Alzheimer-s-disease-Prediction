# 📌 Install required packages if not already installed
#install.packages(c("caret", "randomForest", "glmnet", "ggplot2"))

# 📌 Load necessary libraries
library(caret)
library(randomForest)
library(glmnet)
library(ggplot2)

# 📌 Load dataset
data <- read.csv("balanced_alzheimers_data.csv")

# 📌 Convert target variable to factor
data$Diagnosis <- as.factor(data$Diagnosis)

# 📌 Separate features and target variable
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

# Extract feature importance from Random Forest model used in RFE
rfe_importance_df <- data.frame(Feature = rownames(varImp(rfe_model)$importance),
                                Importance = varImp(rfe_model)$importance$Overall)

# Sort by importance
rfe_importance_df <- rfe_importance_df[order(-rfe_importance_df$Importance), ]

### 🔹 2️⃣ Lasso Regression for Feature Selection
set.seed(42)

# Convert data to matrix format for glmnet
X_matrix <- as.matrix(X)

# Train Lasso model with cross-validation
lasso_model <- cv.glmnet(X_matrix, y, family="binomial", alpha=1)

# Get features with non-zero coefficients
lasso_coef <- coef(lasso_model, s="lambda.min")
lasso_selected_features <- rownames(lasso_coef)[lasso_coef[,1] != 0]
lasso_selected_features <- lasso_selected_features[-1]  # Remove intercept
print("✅ Lasso Selected Features:")
print(lasso_selected_features)

# Get Lasso feature importance scores (absolute coefficient values)
lasso_importance_df <- data.frame(Feature = rownames(lasso_coef), Importance = abs(as.vector(lasso_coef)))
lasso_importance_df <- lasso_importance_df[lasso_importance_df$Importance > 0, ]  # Remove zero coefficients
lasso_importance_df <- lasso_importance_df[order(-lasso_importance_df$Importance), ]  # Sort by importance

### 🔹 3️⃣ Random Forest Feature Importance
set.seed(42)

# Train Random Forest model
rf_model <- randomForest(X, y, ntree=100, importance=TRUE)

# Get feature importance
rf_importance <- importance(rf_model)
rf_importance_df <- data.frame(Feature=rownames(rf_importance), Importance=rf_importance[, "MeanDecreaseGini"])

# Sort and print feature importance scores
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]
print("✅ Random Forest Feature Importance Scores:")
print(rf_importance_df)

### 🔹 Save Results to CSV
write.csv(rfe_importance_df, "RFE_Feature_Importance.csv", row.names=FALSE)
write.csv(lasso_importance_df, "Lasso_Feature_Importance.csv", row.names=FALSE)
write.csv(rf_importance_df, "RandomForest_Feature_Importance.csv", row.names=FALSE)

print("📂 Feature selection results saved to CSV files.")

### 📊 Feature Importance Visualization
# 📌 Random Forest Feature Importance Plot
ggplot(rf_importance_df[1:10,], aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title="Random Forest Feature Importance", x="Features", y="Importance")

# 📌 Lasso Feature Importance Plot
ggplot(lasso_importance_df[1:10,], aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="darkred") +
  coord_flip() +
  theme_minimal() +
  labs(title="Lasso Feature Importance", x="Features", y="Importance")

# 📌 RFE Feature Importance Plot
ggplot(rfe_importance_df[1:10,], aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="purple") +
  coord_flip() +
  theme_minimal() +
  labs(title="RFE Feature Importance", x="Features", y="Importance")

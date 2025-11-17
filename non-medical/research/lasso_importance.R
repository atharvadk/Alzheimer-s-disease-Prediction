if (!require(glmnet)) install.packages("glmnet")
if (!require(caret)) install.packages("caret")

library(glmnet)
library(caret)

set.seed(123)

data <- read.csv("balanced_alzheimers_data.csv")

x <- as.matrix(data[, 1:32])  # All columns except mpg as features
y <- data[, 33]              # mpg as target variable

cv_lasso <- cv.glmnet(x, y, alpha = 1, nfolds = 10)

plot(cv_lasso)

lambda_min <- cv_lasso$lambda.min  # Lambda that gives minimum MSE
lambda_1se <- cv_lasso$lambda.1se  # Lambda within 1 standard error of minimum MSE

cat("Lambda min:", lambda_min, "\n")
cat("Lambda 1se:", lambda_1se, "\n")

lasso_model <- glmnet(x, y, alpha = 1, lambda = lambda_min)

lasso_coef <- coef(lasso_model)
print("LASSO coefficients (using lambda_min):")
print(lasso_coef)

nonzero_coef <- lasso_coef[lasso_coef[, 1] != 0, , drop = FALSE]
cat("\nSelected features (non-zero coefficients):\n")
print(nonzero_coef)

var_importance <- abs(lasso_coef[-1, 1])  # Exclude intercept
names(var_importance) <- colnames(x)

sorted_importance <- sort(var_importance, decreasing = TRUE)
cat("\nFeature importance ranking:\n")
print(sorted_importance)

barplot(sorted_importance, 
        main = "LASSO Feature Importance", 
        horiz = TRUE, 
        col = "steelblue",
        las = 1,
        cex.names = 0.7)
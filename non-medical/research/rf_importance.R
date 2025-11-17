
if (!require(randomForest)) install.packages("randomForest")
if (!require(caret)) install.packages("caret")

library(randomForest)
library(caret)

set.seed(123)

data <- read.csv("balanced_alzheimers_data.csv")

features <- data[, 1:32]  
target <- data[, 33]

if(is.factor(target) || is.character(target)) {
  rf_model <- randomForest(x = features, 
                           y = as.factor(target), 
                           ntree = 500,
                           importance = TRUE)
} else {
  rf_model <- randomForest(x = features, 
                           y = target, 
                           ntree = 500,
                           importance = TRUE)
}

print(rf_model)

importance_scores <- importance(rf_model)
print(importance_scores)

varImpPlot(rf_model, 
           sort = TRUE, 
           main = "Random Forest Feature Importance")

if(is.factor(target) || is.character(target)) {
  imp_df <- data.frame(Feature = rownames(importance_scores),
                       Importance = importance_scores[, "MeanDecreaseGini"])
} 

imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
print(imp_df)

top_n <- 10
top_features <- as.character(imp_df$Feature[1:min(top_n, nrow(imp_df))])
cat("\nTop", min(top_n, nrow(imp_df)), "features:\n")
print(top_features)
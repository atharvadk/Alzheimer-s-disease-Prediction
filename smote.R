
library(smotefamily)
library(tidyverse)


df <- read.csv("alzheimers_disease_data.csv")


df$Diagnosis <- as.factor(df$Diagnosis)


table(df$Diagnosis)

# Apply SMOTE
set.seed(123)
smote_data <- SMOTE(df[, -ncol(df)], df$Diagnosis, K = 5, dup_size = 2)


df_balanced <- smote_data$data
colnames(df_balanced)[ncol(df_balanced)] <- "Diagnosis"


df_balanced$Diagnosis <- as.factor(df_balanced$Diagnosis)


binary_cols <- names(df)[sapply(df, function(x) is.numeric(x) && all(x %in% c(0,1)))]


categorical_cols <- c("Ethnicity", "EducationLevel")


df_balanced[binary_cols] <- lapply(df_balanced[binary_cols], round)
df_balanced[categorical_cols] <- lapply(df_balanced[categorical_cols], round)


df_balanced$Age <- round(df_balanced$Age)


print("Class distribution after SMOTE:")
print(table(df_balanced$Diagnosis))


min_class_count <- min(table(df_balanced$Diagnosis))
df_final <- df_balanced %>%
  group_by(Diagnosis) %>%
  sample_n(min_class_count) %>%
  ungroup()


print("Final balanced class distribution:")
print(table(df_final$Diagnosis))


write.csv(df_final, "alzheimers_balanced_final.csv", row.names = FALSE)

print("Balanced dataset saved as 'alzheimers_balanced_final.csv'.")

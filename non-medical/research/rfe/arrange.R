# Install and load readr if not already installed
if (!require(readr)) install.packages("readr")
library(readr)

# Read your CSV file
data <- read_csv("balanced_alzheimers_data.csv")

# Define your desired column order
#new_order <- c("MemoryComplaints", "BehavioralProblems", "Forgetfulness", 
#               "EducationLevel", "Gender", "Diabetes", "Smoking", 
#              "FamilyHistoryAlzheimers", "Depression", "Age", "SleepQuality", 
#              "CardiovascularDisease", "Confusion", "Disorientation", 
#               "Ethnicity", "Hypertension", "BMI", "DifficultyCompletingTasks", 
#               "PhysicalActivity", "PersonalityChanges", "HeadInjury", 
#              "AlcoholConsumption", "DietQuality", "Diagnosis")


#new_order <- c("MemoryComplaints", "BehavioralProblems", "Forgetfulness", 
#              "EducationLevel", "Gender", "Smoking", "Diabetes", 
#               "FamilyHistoryAlzheimers", "Depression", "SleepQuality", 
#               "CardiovascularDisease", "Age", "Ethnicity", "Confusion", 
#               "Disorientation", "BMI", "Hypertension", "DifficultyCompletingTasks", 
#               "AlcoholConsumption", "PhysicalActivity", "PersonalityChanges", 
#               "HeadInjury", "DietQuality","Diagnosis")


#new_order <- c("MemoryComplaints", "BehavioralProblems", "Hypertension", 
 #              "CardiovascularDisease", "EducationLevel", "HeadInjury", 
  #             "SleepQuality", "Diabetes", "FamilyHistoryAlzheimers", "Age", 
   #            "Diagnosis")

#new_order <- c("FunctionalAssessment", "ADL", "MMSE", "MemoryComplaints", 
 #              "BehavioralProblems", "EducationLevel", "Hypertension", 
  #             "Smoking", "DifficultyCompletingTasks", "CholesterolLDL", 
   #            "SystolicBP", "CholesterolTriglycerides", "Forgetfulness", 
    #           "Diabetes", "SleepQuality", "Gender", "CardiovascularDisease", 
     #         "FamilyHistoryAlzheimers", "Age", "Confusion", 
      #         "PhysicalActivity", "Depression", "Ethnicity", 
       #        "CholesterolHDL", "PersonalityChanges", "DietQuality", 
        #       "CholesterolTotal", "Disorientation", "DiastolicBP", 
         #      "HeadInjury", "BMI", "AlcoholConsumption", "Diagnosis")

#new_order <- c("MemoryComplaints", "BehavioralProblems", "FunctionalAssessment", 
 #              "ADL", "Smoking", "CardiovascularDisease", "Hypertension", 
  #             "HeadInjury", "MMSE", "FamilyHistoryAlzheimers", "Confusion", 
   #            "SleepQuality", "EducationLevel", "Disorientation", "Ethnicity", 
    #           "Depression", "AlcoholConsumption", "Age", "CholesterolHDL", 
     #          "CholesterolLDL", "CholesterolTriglycerides", "Gender", "BMI", 
      #        "DiastolicBP", "CholesterolTotal", "PersonalityChanges", 
       #        "DifficultyCompletingTasks", "Forgetfulness", "Diagnosis")

new_order <- c("MemoryComplaints", "BehavioralProblems", "Hypertension", "CardiovascularDisease", 
               "EducationLevel", "SleepQuality", "Diabetes", "FamilyHistoryAlzheimers", 
               "Age", "Forgetfulness", "DifficultyCompletingTasks", "Disorientation", 
               "PersonalityChanges", "FunctionalAssessment", "ADL", "MMSE", 
               "Smoking", "CholesterolTriglycerides", "CholesterolHDL", "Gender", 
               "SystolicBP", "Depression", "CholesterolTotal", "CholesterolLDL", 
               "PhysicalActivity", "Diagnosis")




# Check if all new_order columns exist in the dataset
missing_cols <- setdiff(new_order, colnames(data))
if (length(missing_cols) > 0) {
  warning("The following columns in new_order are not in the dataset: ", 
          paste(missing_cols, collapse = ", "))
}

# Check if all original columns are included in new_order
extra_cols <- setdiff(colnames(data), new_order)
if (length(extra_cols) > 0) {
  warning("The following columns in the dataset are not in new_order: ", 
          paste(extra_cols, collapse = ", "))
}

# Rearrange columns (only include columns that exist in both)
data_reordered <- data[, intersect(new_order, colnames(data))]

# Check the new column order
print(colnames(data_reordered))

# Write the reordered data to a new CSV file
write_csv(data_reordered, "rfe_columns.csv")
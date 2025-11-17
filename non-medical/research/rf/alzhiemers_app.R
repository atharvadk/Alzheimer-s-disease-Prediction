
if (!require("readr")) install.packages("readr")
if (!require("adabag")) install.packages("adabag")
if (!require("shiny")) install.packages("shiny")


library(readr)
library(adabag)
library(shiny)
data <- read_csv("rf_nonMedical_columns.csv")


print(colnames(data))

# Select first 8 features and the target
features <- data[, 1:8]
target <- data$Diagnosis  

# Combine features + target
df <- cbind(features, Diagnosis = as.factor(target))

# Train AdaBoost model
set.seed(123)
model <- boosting(Diagnosis ~ ., data = df)

# UI
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
      }
      .card {
        background-color: beige;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
      }
      h1, h3 {
        color: #2c3e50;
      }
      .result-text {
        font-size: 20px;
        font-weight: bold;
        color: #4e73df;
        padding-top: 10px;
      }
    "))
  ),
  
  titlePanel("🧠 Alzheimer's Disease Detection"),
  
  fluidRow(
    column(12,
           div(class = "card",
               strong("Note: For all fields, unless specified otherwise — 0 = No, 1 = Yes.")
           )
    )
  ),
  
  fluidRow(
    column(4,
           div(class = "card",
               lapply(1:8, function(i) {
                 label <- paste(colnames(df)[i])
                 input_id <- paste0("feature", i)
                 help_text <- ""
                 choices <- c("0", "1")
                 
                 # Custom dropdowns with special explanations
                 if (grepl("educat|educ|education", tolower(label))) {
                   help_text <- "0 = None, 1 = High School, 2 = Bachelors, 3 = Higher"
                   choices <- c("0", "1", "2", "3")
                 } else if (grepl("gender|sex", tolower(label))) {
                   help_text <- "0 = Male, 1 = Female"
                   choices <- c("0", "1")
                 }
                 
                 tagList(
                   selectInput(inputId = input_id, label = label, choices = choices, selected = "0"),
                   if (help_text != "") helpText(help_text)
                 )
               }),
               actionButton("predictBtn", "🔍 Predict", class = "btn btn-primary")
           )
    ),
    
    column(8,
           div(class = "card",
               h3("Prediction Result:"),
               div(textOutput("prediction"), class = "result-text")
           )
    )
  )
)

# Server
server <- function(input, output) {
  observeEvent(input$predictBtn, {
    new_data <- data.frame(
      matrix(unlist(lapply(1:8, function(i) as.numeric(input[[paste0("feature", i)]]))), nrow = 1)
    )
    colnames(new_data) <- colnames(df)[1:8]
    
    prediction <- predict.boosting(model, new_data)
    pred_class <- prediction$class
    
    message <- if (pred_class %in% c("Yes", "1", "Alzheimer")) {
      "🧠 You are likely to have Alzheimer's disease."
    } else {
      "✅ You are unlikely to have Alzheimer's disease."
    }
    
    output$prediction <- renderText({ message })
  })
}


shinyApp(ui = ui, server = server)

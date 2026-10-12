# install.packages(c("mlbench", "caret"))
library(mlbench)
library(caret)

data(BreastCancer)
df <- BreastCancer
df$Id <- NULL
df <- na.omit(df)                       # remove missing values

# Convert predictors (factors) to numeric
for (i in 1:9) df[, i] <- as.numeric(as.character(df[, i]))

set.seed(123)
idx   <- createDataPartition(df$Class, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

# Logistic regression model
model <- glm(Class ~ ., data = train, family = binomial)
summary(model)

# Predict probability of malignant, then classify
prob <- predict(model, newdata = test, type = "response")
pred <- factor(ifelse(prob > 0.5, "malignant", "benign"),
               levels = levels(test$Class))

# Evaluation
confusionMatrix(pred, test$Class, positive = "malignant")

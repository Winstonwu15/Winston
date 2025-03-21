bank_data <- read.csv("bank_personal_loan.csv")

head(bank_data)

str(bank_data)

summary(bank_data)

colSums(is.na(bank_data))

library(ggplot2)

ggplot(bank_data, aes(x = Age)) + 
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Age Distribution", x = "Age", y = "Frequency")

ggplot(bank_data, aes(x = CreditCard)) +
  geom_bar(fill = "orange", color = "black") +
  theme_minimal() +
  labs(title = "CreditCard Ownership Distribution", x = "CreditCard", y = "Count")

ggplot(bank_data, aes(x = Income)) +
  geom_histogram(binwidth = 10, fill = "green", color = "black") +
  theme_minimal() +
  labs(title = "Income Distribution", x = "Income", y = "Frequency")
set.seed(123)
library(caret)
split <- createDataPartition(bank_data$CreditCard, p = 0.7, list = FALSE)
train_data <- bank_data[split, ]
test_data <- bank_data[-split, ]

train_data$Personal.Loan <- as.factor(train_data$Personal.Loan)
test_data$Personal.Loan <- as.factor(test_data$Personal.Loan)

num_cols <- c("Age", "Experience", "Income", "Family", "CCAvg", "Mortgage")
train_data[num_cols] <- scale(train_data[num_cols])
test_data[num_cols] <- scale(test_data[num_cols])

train_data$Personal.Loan <- factor(train_data$Personal.Loan, levels = c(0, 1))
test_data$Personal.Loan <- factor(test_data$Personal.Loan, levels = c(0, 1))

logistic_model <- glm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage, 
                      data = train_data, family = binomial)

logistic_pred <- predict(logistic_model, test_data, type = "response")

logistic_pred_class <- ifelse(logistic_pred > 0.5, 1, 0)

logistic_pred_class <- factor(logistic_pred_class, levels = levels(test_data$Personal.Loan))

logistic_conf_matrix <- confusionMatrix(logistic_pred_class, test_data$Personal.Loan)
logistic_conf_matrix

library(e1071)

svm_model <- svm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage,
                 data = train_data, probability = TRUE)

svm_pred <- predict(svm_model, test_data)

svm_pred_class <- factor(svm_pred, levels = levels(test_data$Personal.Loan))

svm_conf_matrix <- confusionMatrix(svm_pred_class, test_data$Personal.Loan)
svm_conf_matrix

library(randomForest)

rf_model <- randomForest(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage,
                         data = train_data, ntree = 100)

rf_pred <- predict(rf_model, test_data)

rf_pred_class <- factor(rf_pred, levels = levels(test_data$Personal.Loan))

rf_conf_matrix <- confusionMatrix(rf_pred_class, test_data$Personal.Loan)
rf_conf_matrix

library(pROC)

logistic_pred_prob <- predict(logistic_model, newdata = test_data, type = "response")

rf_pred_prob <- predict(rf_model, newdata = test_data, type = "prob")[, 2] 

svm_pred_prob <- predict(svm_model, newdata = test_data, probability = TRUE)
svm_pred_prob <- attributes(svm_pred_prob)$probabilities[, 2]

logistic_roc <- roc(test_data$Personal.Loan, logistic_pred_prob)
rf_roc <- roc(test_data$Personal.Loan, rf_pred_prob)
svm_roc <- roc(test_data$Personal.Loan, svm_pred_prob)

plot(logistic_roc, col = "blue", main = "ROC Curve for Logistic Regression, Random Forest, and SVM")
plot(rf_roc, col = "red", add = TRUE)
plot(svm_roc, col = "green", add = TRUE)

legend("bottomright", legend = c("Logistic Regression", "Random Forest", "SVM"),
       col = c("blue", "red", "green"), lwd = 2)

library(caret)

ctrl <- trainControl(method = "cv", number = 5)

logistic_model_cv <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage,
                           data = train_data, method = "glm", family = "binomial", trControl = ctrl)

svm_model_cv <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage,
                      data = train_data, method = "svmLinear", trControl = ctrl)

rf_model_cv <- train(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage,
                     data = train_data, method = "rf", trControl = ctrl)

print(logistic_model_cv)
print(svm_model_cv)
print(rf_model_cv)


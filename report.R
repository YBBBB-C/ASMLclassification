#run the chunks seperately as it cxontains data check frm time to time 

---


library(tidyverse)
library(caret)
library(xgboost)
library(ROSE)
library(pROC)
library(dplyr)


data <- read.csv("bank_personal_loan.csv")
str(data)

# repeat term
cat("record the repeating number：", sum(duplicated(data)), "\n")
data <- data %>% distinct()  # remove the duplicate

# missing value
cat("feature missing stats：\n")
colSums(is.na(data))

# outlier value
summary(data)

# Personal Loan category ratio
prop.table(table(data$Personal.Loan))

# Correlation analysis, numerical features only
numeric_features <- data %>% 
  select(-ZIP.Code, -Personal.Loan) # remove zip and target

cor_matrix <- cor(numeric_features)


print(cor_matrix)
library(corrplot)
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "lightcoral",tl.cex = 0.8, number.cex = 0.7)


# remove  Experience as it is highly correlated with age
data <- data %>% select(-Experience)
```

#3 EDA analysis
library(ggplot2)

# Personal Loan distribution bar chart
ggplot(data, aes(x = factor(Personal.Loan))) +
  geom_bar(fill = c("skyblue", "salmon")) +
  labs(title = "Personal Loan Distribution", x = "Personal Loan", y = "Count")

# Income vs Personal Loan
ggplot(data, aes(x = factor(Personal.Loan), y = Income)) +
  geom_boxplot(fill = c("skyblue", "salmon")) +
  labs(title = "Income vs Personal Loan")

# CCAvg vs Personal Loan
ggplot(data, aes(x = factor(Personal.Loan), y = CCAvg)) +
  geom_boxplot(fill = c("skyblue", "salmon")) +
  labs(title = "CCAvg vs Personal Loan")

# CD Account vs Personal Loan
table_CD <- as.data.frame(table(data$CD.Account, data$Personal.Loan))
colnames(table_CD) <- c("CD.Account", "Personal.Loan", "Count")

ggplot(table_CD, aes(x = Personal.Loan, y = Count, fill = factor(CD.Account))) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  labs(title = "CD Account vs Personal Loan", x = "Personal Loan", y = "Count", fill = "CD Account") +
  theme_minimal()


# delet ZIP Code
data <- data %>% select(-ZIP.Code)



#Feature Engineering (factor processing + standardized scale + SMOTE processing category imbalance)
library(caret)
library(DMwR2) 
library(ROSE)   
library(dplyr)

# 1. factor transformation
data_factor <- data %>%
  mutate(across(c(Education, Securities.Account, CD.Account, Online, CreditCard, Family), as.factor))

# 2. ROSE Handle category imbalance (handle imbalance first, then standardize)
set.seed(123)  
data_balanced <- ROSE(Personal.Loan ~ ., data = data_factor, seed = 123)$data
data_balanced_scaled <- data_balanced %>%
  mutate(across(c(Age, Income, CCAvg, Mortgage), scale))
data_balanced_scaled$Personal.Loan <- as.factor(data_balanced_scaled$Personal.Loan)

prop.table(table(data_balanced_scaled$Personal.Loan))



#data division
set.seed(123) 
trainIndex <- createDataPartition(data_balanced_scaled$Personal.Loan, p = 0.7, list = FALSE)
train_data <- data_balanced_scaled[trainIndex, ]
test_data <- data_balanced_scaled[-trainIndex, ]


table(train_data$Personal.Loan)
table(test_data$Personal.Loan)




str(data_balanced_scaled)  # check again all variabls are still there

#Check again before training 
colSums(is.na(data_balanced_scaled))
#no missing value if all outcome is 0
prop.table(table(data_balanced_scaled$Personal.Loan))
#category imbalance is addressed，it should return a fifty-fifty result

str(data_balanced_scaled)

summary(select(data_balanced_scaled, Age, Income, CCAvg, Mortgage))
apply(select(data_balanced_scaled, Age, Income, CCAvg, Mortgage), 2, sd)
#Check the statistical information of the standardized numerical variables to ensure that the mean is close to 0 and the standard deviation is close to 1:

sum(is.na(train_data$Personal.Loan))
sum(is.na(test_data$Personal.Loan))


# Method 1
library(randomForest)

set.seed(123)
# random forest(loss = Gini Impurity)
rf_model <- randomForest(Personal.Loan ~ ., data=train_data, importance=TRUE)

print(rf_model)
rf_pred_prob <- predict(rf_model, test_data, type="prob")[,2]
rf_pred <- predict(rf_model, test_data, type="response")

# confusion matrix and recall
confusionMatrix(rf_pred, test_data$Personal.Loan, positive="1")

# ROC curve & AUC
rf_roc <- roc(as.numeric(test_data$Personal.Loan), rf_pred_prob)
plot(rf_roc, col = "lightcoral", main = "ROC Curve - Random Forest")
cat("Random Forest ROC-AUC:", auc(rf_roc), "\n")

# PR-AUC curve
rf_pr <- pr.curve(scores.class0 = rf_pred_prob[test_data$Personal.Loan=="1"],
                  scores.class1 = rf_pred_prob[test_data$Personal.Loan=="0"],
                  curve = TRUE)
plot(rf_pr, main="PR Curve - Random Forest")
cat("Random Forest PR-AUC:", rf_pr$auc.integral, "\n")

# feature importance
importance(rf_model)
varImpPlot(rf_model, main="Feature Importance - Random Forest")



sum(is.na(train_data))  
sum(is.na(test_data))

#method 2
library(xgboost)

set.seed(123)

# data praperation
train_matrix <- model.matrix(Personal.Loan~.-1, train_data)
train_label <- as.numeric(train_data$Personal.Loan) - 1

test_matrix <- model.matrix(Personal.Loan~.-1, test_data)
test_label <- as.numeric(test_data$Personal.Loan) - 1

dtrain <- xgb.DMatrix(data=train_matrix, label=train_label)
dtest <- xgb.DMatrix(data=test_matrix, label=test_label)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# early stopping rounds
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train=dtrain, eval=dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

# predict
xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, "1", "0") %>% factor(levels = c("0","1"))

confusionMatrix(xgb_pred, test_data$Personal.Loan, positive="1")

xgb_roc <- roc(test_label, xgb_pred_prob)
plot(xgb_roc, col = "lightblue", main = "ROC Curve - XGBoost")
cat("XGBoost ROC-AUC:", auc(xgb_roc), "\n")

xgb_pr <- pr.curve(scores.class0 = xgb_pred_prob[test_label==1],
                   scores.class1 = xgb_pred_prob[test_label==0],
                   curve = TRUE)
plot(xgb_pr, main="PR Curve - XGBoost")
cat("XGBoost PR-AUC:", xgb_pr$auc.integral, "\n")

importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix)


rf_roc <- roc(as.numeric(test_data$Personal.Loan), rf_pred_prob)
xgb_roc <- roc(as.numeric(test_data$Personal.Loan), xgb_pred_prob)


plot(rf_roc, col = "lightcoral", lwd = 2, main = "ROC Curve Comparison")
lines(xgb_roc, col = "lightblue", lwd = 2, lty = 2)

legend("bottomright", legend = c("Random Forest", "XGBoost"),
       col = c("lightcoral", "lightblue"), lwd = 2,lty = c(1, 2))

cat("Random Forest ROC-AUC:", auc(rf_roc), "\n")
cat("XGBoost ROC-AUC:", auc(xgb_roc), "\n")

rf_pr <- pr.curve(scores.class0 = rf_pred_prob[which(test_data$Personal.Loan == 0)],
                  scores.class1 = rf_pred_prob[which(test_data$Personal.Loan == 1)],
                  curve = TRUE)

xgb_pr <- pr.curve(scores.class0 = xgb_pred_prob[which(test_data$Personal.Loan == 0)],
                   scores.class1 = xgb_pred_prob[which(test_data$Personal.Loan == 1)],
                   curve = TRUE)


plot(rf_pr$curve, col = "lightcoral", type = "l", lwd = 2, xlab = "Recall", ylab = "Precision", main = "PR Curve Comparison")
lines(xgb_pr$curve, col = "lightblue", lwd = 2, lty = 2)


legend("bottomleft", legend = c("Random Forest", "XGBoost"),
       col = c("lightcoral", "lightblue"), lwd = 2,lty = c(1, 2))


cat("Random Forest PR-AUC:", rf_pr$auc.integral, "\n")
cat("XGBoost PR-AUC:", xgb_pr$auc.integral, "\n")

#choose xgboost because it is more stable. Now process hyperparameter tuning 
library(caret)
library(xgboost)
train_data$Personal.Loan <- factor(train_data$Personal.Loan, levels = c(0,1), labels = c("No", "Yes"))
test_data$Personal.Loan <- factor(test_data$Personal.Loan, levels = c(0,1), labels = c("No", "Yes"))


tune_grid <- expand.grid(
  max_depth = c(3, 4, 5), 
  eta = c(0.01, 0.05, 0.1), 
  nrounds = c(100, 300),
  subsample = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.7, 0.8, 0.9),
  min_child_weight = c(1, 3, 5),
  gamma = 0  
)

# 
train_control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,  
  summaryFunction = twoClassSummary,  
  verboseIter = TRUE  
)

# train XGBoost 
set.seed(123)
xgb_tuned <- train(
  Personal.Loan ~ ., 
  data = train_data,
  method = "xgbTree",
  trControl = train_control, 
  tuneGrid = tune_grid,
  
)

# best pram
print(xgb_tuned)

library(xgboost)
library(caret)
library(pROC)
library(PRROC)

set.seed(123)


train_matrix <- model.matrix(Personal.Loan~.-1, train_data)
train_label <- as.numeric(train_data$Personal.Loan) - 1

test_matrix <- model.matrix(Personal.Loan~.-1, test_data)
test_label <- as.numeric(test_data$Personal.Loan) - 1

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)


params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.03,            
  max_depth = 6,         
  subsample = 0.8,       
  colsample_bytree = 0.7, 
  min_child_weight = 3, 
  lambda = 1,           
  alpha = 0.5          
)


xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(eval = dtest),
  early_stopping_rounds = 10,
  verbose = 0  
)


xgb_pred_prob <- predict(xgb_model, dtest)

xgb_pred <- factor(ifelse(xgb_pred_prob > 0.4, "1", "0"), levels = c("0", "1"))

test_data$Personal.Loan <- as.factor(test_data$Personal.Loan)



#confusionMatrix(xgb_pred, test_data$Personal.Loan, positive="1")

```


```{r}

xgb_roc <- roc(test_label, xgb_pred_prob)
plot(xgb_roc, col = "lightblue", main = "ROC Curve - XGBoost")
cat("XGBoost ROC-AUC:", auc(xgb_roc), "\n")


xgb_pr <- pr.curve(scores.class0 = xgb_pred_prob[test_label == 1],
                   scores.class1 = xgb_pred_prob[test_label == 0],
                   curve = TRUE)
plot(xgb_pr, main = "PR Curve - XGBoost")
cat("XGBoost PR-AUC:", xgb_pr$auc.integral, "\n")


importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix)
```


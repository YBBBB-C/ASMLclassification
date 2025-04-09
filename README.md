# ASMLclassification
##if you are grader,pls ignore the readme, this is for personal project showing use.

This analysis aimed to help the bank predict whether a customer would accept a personal loan
by analysing various customer data points. The dataset contained multiple factors, such as
income, family size, credit card usage, and education level, which might influence loan
acceptance. Before building our models, we first cleaned the data by removing duplicates,
handling missing values, and ensuring that only relevant and accurate information was used.
This step was crucial in making sure that our models learned from high-quality and
meaningful data.

To find the best predictive model, we tested two machine learning models: Random Forest
and XGBoost. We compared their performance using different evaluation metrics and
ultimately selected XGBoost because it performed better overall. A key concern in this
prediction task was minimizing two types of errors:
False Negatives (missed detections): Customers who would have accepted the loan but were
incorrectly predicted as uninterested. This could cause the bank to miss potential business
opportunities. False Positives (incorrect detections): Customers who were predicted to accept
the loan but would not. This could lead to wasted marketing efforts and unnecessary costs
We analysed two key evaluation methods: one is Confusion Matrix – A table comparing
actual outcomes with model predictions, helping us see how often the model misclassifies
customers.

XGBoost achieved an accuracy of 94.46%, meaning it correctly classified most customers.
The false negative rate was 6.46%, significantly lower than Random Forest’s 6.86%, showing
that XGBoost missed fewer potential loan accepters. The false positive rate was 4.63%,
slightly better than Random Forest’s 5.37%, meaning fewer customers were incorrectly
predicted to accept the loan.

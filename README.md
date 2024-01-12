This project aims to develop a robust model for predicting whether the District Department of Transportation in Washington D.C. should increase the number of bikes available in the public bicycle-sharing system during specific hours. This prediction is based on various factors, including date and time, weather conditions, temperature, and other relevant features. The task is treated as a binary classification problem. The training dataset comprises 1600 instances of hourly bike rental features, and a test set of 400 instances will be provided for evaluation.

Initially, an examination of the training data was conducted to identify any significant trends in bicycle demand from different perspectives:
- Are there discernible trends when comparing different hours, weeks, and months?
- Is there a distinction between weekdays and holidays?
- Does bicycle demand exhibit any trends based on weather conditions, such as rainy or snowy days?

An analysis of outliers was performed using box plots to identify potential outliers. Subsequently, a qualitative study was conducted on these data points to determine whether they should be excluded from the dataset.

Feature selection was then carried out to determine the features to include in the model. The correlation with bike demand was examined, utilizing the chi-squared test for categorical values and the one-way ANOVA test for numerical features.

Following feature selection, a random forest classifier was trained on the training data split of the dataset using the scikit-learn library. A hyperparameter study was undertaken to identify optimal parameters. This study involved a grid search of max_features and min_samples_leaf, with evaluation performed using a 5-fold stratified cross-validation. The F1-score metric was employed due to the dataset's significant imbalance.

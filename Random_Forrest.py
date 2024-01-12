import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

## Import training data
dataSet = pd.read_csv('training_data.csv')

## Adding new features
snowy = []
for i in dataSet['snowdepth']:
    if i > 0:
        snowy.append(1)
    else:
        snowy.append(0)
dataSet["snowy"] = snowy

## Outliers, visibility
indexOutliers = dataSet[(dataSet['visibility'] <= 2)].index
print('Outliers:', len(dataSet[(dataSet['visibility'] <= 2)]))
dataSet = dataSet.drop(index=indexOutliers)

## Outliers, windspeed
indexOutliers = dataSet[(dataSet['windspeed'] >= 33)].index
print('Outliers:', len(dataSet[(dataSet['windspeed'] >= 33)]))
dataSet = dataSet.drop(labels=indexOutliers)

dataSet = dataSet.drop(columns=['snow', 'snowdepth', 'dew', 'holiday'])

## Mapping Increase_Stock to have binary values
dataSet['increase_stock'] = dataSet['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Move Increase_Stock to the rightmost column in the dataset
column_to_move = dataSet.pop('increase_stock')
dataSet['increase_stock'] = column_to_move


#### Splitting data
np.random.seed(1)

ratio = 0.75    # percentage of dataset to be training data, the rest becomes the test set
print(ratio*100, "% of test data =", int(len(dataSet)*ratio))
trainIndex = np.random.choice(dataSet.shape[0], size=int(len(dataSet)*.75), replace=False) # randomly choose a number of indices
trainIndexBool = dataSet.index.isin(trainIndex) # create an array with true/false based on whether the index is in the dataset
train = dataSet.iloc[trainIndex]                # select training data from the locations in trainIndex that are True
test = dataSet.iloc[~trainIndexBool]            # the remaining becomes test data

xTrain = train.copy().drop(columns=['increase_stock'])
yTrain = train['increase_stock']
xTest = test.copy().drop(columns=['increase_stock'])
yTest = test['increase_stock']


#### Hyperparameter Grid
n_estimators = [2000] # Number of trees in the random forest

max_features = [int(x) for x in np.linspace(4, 13, 10)] # Number of features to consider at every split

min_samples_leaf = [int(x) for x in np.linspace(1, 10, 10)] # Minimum number of samples required at each leaf node

bootstrap = [True] # Method of selecting samples for training each tree

# Create the hyperparameter grid
paramGrid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}

#### Grid Search
modelForrest = RandomForestClassifier(random_state=0)

search = GridSearchCV(estimator = modelForrest, 
                        param_grid = paramGrid, 
                        cv = 5,
                        verbose = 2,
                        n_jobs = -1,
                        return_train_score = True,
                        scoring='f1_macro')

search.fit(xTrain, yTrain)

# Create variables for the best model
bestModelForrest = search.best_estimator_
scores = search.cv_results_


#### Print best hyperparameters and evaluation
print('Best average CV score:', search.best_score_)
print('Best parameters:', search.best_params_)

data = {
    'param_max_features': list(search.cv_results_['param_max_features'].data),
    'param_min_samples_leaf': list(search.cv_results_['param_min_samples_leaf'].data),
    'mean_test_score': list(search.cv_results_['mean_test_score'])
}
df = pd.DataFrame(data)

# Pivot the DataFrame
pivot_df = df.pivot(index='param_min_samples_leaf', columns='param_max_features', values='mean_test_score')

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=".3f", linewidths=.5)
plt.title('Grid Search High demand F1 Macro Heatmap')
plt.show()

#### Train model
paramCopy = search.best_params_.copy()
paramCopy['n_estimators'] = 10000

modelForrest = RandomForestClassifier(**paramCopy, random_state=0, n_jobs=-1,)

modelForrest.fit(xTrain, yTrain)

# Print the best hyperparameters
yPredict = modelForrest.predict(xTest)
print(skmet.classification_report(y_true=yTest, y_pred=yPredict))


#### Predicting on the prduction data
## Import training data
dataSet = pd.read_csv('test_data.csv')

## Adding new features
snowy = []
for i in dataSet['snowdepth']:
    if i > 0:
        snowy.append(1)
    else:
        snowy.append(0)
dataSet["snowy"] = snowy

dataSet = dataSet.drop(columns=['dew', 'holiday', 'snow', 'snowdepth'])

## Prediction
yPredict = modelForrest.predict(dataSet)

# Save the DataFrame to a CSV file
predictions_df = pd.DataFrame([yPredict])
predictions_df.to_csv('production_data.csv', index=False, header=False)

print('Number of predictions:', len(yPredict))

import pandas as pd
import numpy as np
import sklearn.metrics as skmet


## Import training data
dataSet = pd.read_csv('Projekt\\training_data.csv')

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


#### Naive model
prob = len(dataSet[dataSet['increase_stock'] == 0])/len(dataSet['increase_stock'])
yNaiveDistributed = np.random.choice([0, 1], size=len(xTest), p=[prob, 1-prob])

print('Naive Distributed:\n', skmet.classification_report(y_true=yTest, y_pred=yNaiveDistributed))
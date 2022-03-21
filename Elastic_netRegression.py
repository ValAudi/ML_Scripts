# CIS 631 Project Using Elastic net Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# plot of learning curve

def plot_learning_curves(best_model, x_train, y_train, x_test, y_test):
    train_errors, val_errors = [], []
    for m in range(1, len(x_train)):
        best_model.fit(x_train[:m], y_train[:m])
        y_train_predict = best_model.predict(x_train[:m])
        y_test_predict = best_model.predict(x_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_test_predict, y_test))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=0.1, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=0.5, label="test")
    plt.show()

# load the dataset. Remove the '#' to select a dataset of choice. Only one can be selected at a time.

dataframe = pd.read_csv("<Datasource.csv>")

# split the dataset for training and testing. 
# 80% is used for trainig and 20 percent is used for testing. 
# The data in each category are chosen randomly.

train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)

x_train, x_test = train_set.iloc[:,5:10], test_set.iloc[:,5:10]
y_train, y_test = train_set.iloc[:,-1:], test_set.iloc[:,-1:]

# In case of any nulls in the dataset fill the null positions with the mean value.

x_test = x_test.fillna(x_train.mean())
x_train = x_train.fillna(x_train.mean())
y_train = y_train.fillna(y_train.mean())

# Machine learning model Elastic Net using early stopping to make it converge

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.1)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(100):
    elastic_net.fit(x_train, y_train.values.ravel())
    y_prediction = elastic_net.predict(x_test)
    val_error = mean_squared_error(y_prediction, y_test)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(elastic_net)

# Get the accuracy of the model

print("Mean Absolute Error: %.2f " % metrics.mean_absolute_error(y_test, y_prediction))
print("Mean Squared Error: %.2f " % metrics.mean_squared_error(y_test, y_prediction))
print("Root Mean Squared Error: %.2f " % np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
print("Model Accuracy: %.2f " % metrics.r2_score(y_test, y_prediction))

# Making a scatter plot of the results
    
df=pd.DataFrame({ 'x':x_test['pop_density'].values.ravel(), 'Actual':y_test.values.ravel(), 'Predicted':y_prediction})
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax1 = df.plot(kind='scatter', x='x', y='Actual', color='r', label='Actual')
ax2 = df.plot(kind='scatter', x='x', y='Predicted', color='g', label='Predicted', ax=ax1)
plt.ylim(0, 5000)
plt.xlim(0.,1000)
plt.legend()
print(ax1 == ax2)
plt.show()
# Uncomment line below to plot the learning curve

# plot_learning_curves(elastic_net, x_train, y_train, x_test, y_test)

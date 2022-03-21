# CIS 631 Project Using Ensemble Regressor

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import VotingRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Plotting learning curve function

# def plot_learning_curves(voting_reg, x_train, y_train, x_test, y_test):
#    train_errors, val_errors = [], []
#    for m in range(1, len(x_train)):
#        voting_reg.fit(x_train[:m], y_train.values.ravel()[:m])
#        y_train_predict = voting_reg.predict(x_train[:m])
#        y_test_predict = voting_reg.predict(x_test)
#        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
#        val_errors.append(mean_squared_error(y_test_predict, y_test))
#    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
#    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="test")
#    plt.show()

# load the dataset. Remove the '#' to select a dataset of choice. Only one can be selected at a time.


dataframe = pd.read_csv("<Datasource.csv>")


# split the dataset for training and testing. 
# 80% is used for trainig and 20 percent is used for testing. 
# The data in each category are chosen randomly.

train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)

# split into input (X) independent and output (y) dependent variables. 
# Different datasets have different choices for input and output variable.

x_train, x_test = train_set.iloc[:,5:10], test_set.iloc[:,5:10]
y_train, y_test = train_set.iloc[:,-1:], test_set.iloc[:,-1:]

# In case of any nulls in the dataset fill the null positions with the mean value.

x_test = x_test.fillna(x_train.mean())
x_train = x_train.fillna(x_train.mean())
y_test = y_test.fillna(y_train.mean())
y_train = y_train.fillna(y_train.mean())

# Machine learning methods in the Ensemble method. 
# The first three are chosen based on their good overall accuracy.

lin_reg = LinearRegression()
rnd_reg = RandomForestRegressor()
svm_reg = LinearSVR(epsilon=0.5)
# sgd_reg = SGDRegressor(penalty='l2'or penalty='l1') for Rigde regession and Lasso regression

# Voting mechanism of the ensemble method

voting_reg = VotingRegressor(estimators=[('lr', lin_reg), ('rf', rnd_reg), ('svr', svm_reg)]) # , ('sgd', sgd_reg)
voting_reg.fit(x_train, y_train.values.ravel())



for reg in (lin_reg, rnd_reg, svm_reg, voting_reg):#  sgd_reg
    reg.fit(x_train, y_train.values.ravel())
    y_prediction = reg.predict(x_test)
    
    print(reg.__class__.__name__, "Mean Absolute Error: %.2f " % metrics.mean_absolute_error(y_test, y_prediction))
    print(reg.__class__.__name__, "Mean Squared Error %.2f " % metrics.mean_squared_error(y_test, y_prediction))
    print(reg.__class__.__name__, "Root Mean Squared Error: %.2f " % np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
    print(reg.__class__.__name__, "R2 Score: %.2f " % metrics.r2_score(y_test, y_prediction))
    
    # Making a scatter plot of the results
    
    df=pd.DataFrame({'x':x_test['pop_density'].values.ravel(), 'Actual':y_test.values.ravel(), 'Predicted':y_prediction})
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    ax1 = df.plot(kind='scatter', x='x', y='Actual', color='r', label='Actual')
    ax2 = df.plot(kind='scatter', x='x', y='Predicted', color='g', label='Predicted', ax=ax1)
    plt.ylim(0, 5000)
    plt.xlim(0.,1000)
    plt.legend()
    print(ax1 == ax2)
    plt.show()

# plot_learning_curves(rnd_reg, x_train, y_train, x_test, y_test)






# CIS 631 Project 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Plotting learning curve function. Uncomment to use

# def plot_learning_curves(model, x_train, y_train, x_test, y_test):
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

x_train, x_test = train_set.iloc[:,5:10], test_set.iloc[:,5:10]
y_train, y_test = train_set.iloc[:,-1:], test_set.iloc[:,-1:]

# In case of any nulls in the dataset fill the null positions with the mean value.

x_test = x_test.fillna(x_train.mean())
x_train = x_train.fillna(x_train.mean())
y_train = y_train.fillna(y_train.mean())

# Defining the Neural Networks Model to use. Input dimensions (Number of dependent variables), 
# Hidden layer and number of nodes, output layer
model = Sequential()

model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# compile the neural network model and set metrics as well as an optimizer for the loss function

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Train the models using the chosen dataset and validate it accuracy score

model.fit(x_train, y_train, epochs=100, validation_split=0.1, batch_size=10)
y_prediction = model.predict(x_test)

# print the R2 score of the model

print("Mean Absolute Error: %.2f " % metrics.mean_absolute_error(y_test, y_prediction))
print("Mean Squared Error: %.2f " % mean_squared_error(y_test, y_prediction))
print("Root Mean Squared Error: %.2f " % np.sqrt(mean_squared_error(y_test, y_prediction)))
print("Model Accuracy: %.2f " % r2_score(y_test, y_prediction))

# Making a scatter plot of the results
    
df=pd.DataFrame({'x':x_test['pop_density'].values.ravel(), 'Actual':y_test.values.ravel(), 'Predicted':y_prediction.ravel()})
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

# plot_learning_curves(model, x_train, y_train, x_test, y_test)

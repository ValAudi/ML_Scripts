#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the pandas, numpy and skitlearn libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the dataset and assign into varible name ('covoid')
covid = pd.read_csv("F:/IT 631/Reserch data/Test.csv")

# view of the loaded data set
covid.head()


# In[3]:


# describing the rows and collumns in the data table
covid.shape


# In[4]:


# describing the main statical values  in data set
covid.describe()


# In[5]:


# get average pop_density into v
v = covid['pop_density'].mean() 

# checking the data set have eny missing values
covid.isnull().any()

# missing value replace with mean
covid = covid.fillna(v) 

# again checking the data set after retrive to find data set have eny missing values
covid.isnull().any()


# In[ ]:





# In[6]:


# dropinf the ('nine','State','County Name') the dats sets and assign other data to ('x') variable 
X = covid.drop(['State','County Name','nine'], axis=1)

# Assign the ('nine') column data set in to ('y') variable
y = covid['nine']


# In[7]:


# importing train_test_split methord from skitlearn and split the data to train and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[8]:


# impoerting decision tree regressor from skitlearn library
from sklearn.tree import DecisionTreeRegressor

# assign ('regressoe') variable to decisiontree regressor.
regressor = DecisionTreeRegressor()

# fitting the data set to model
regressor.fit(X_train, y_train)


# In[9]:


# orediction result assign to ('y_pred') variable
y_pred = regressor.predict(X_test)


# In[10]:


# importing pandas libraries and print the actual and predicted values
import pandas as pd
df=pd.DataFrame({ 'x':X_test['pop_density'] ,'Actual':y_test, 'Predicted':y_pred})
df


# In[15]:


#fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
ax1 = df.plot(kind='scatter', x='x', y='Actual', color='r', label='Actual')    
ax2 = df.plot(kind='scatter', x='x', y='Predicted', color='g', label='Predicted', ax=ax1)
plt.ylim(0, 5000)
plt.xlim(0.,1000)
plt.legend()
print(ax1 == ax2)
plt.show()


# In[30]:



import seaborn as sns
sns.scatterplot(data=df, y="Actual", x="x")
sns.scatterplot(data=df, y="Predicted", x="x")


# In[20]:



import seaborn as sns
sns.scatterplot(y_test,y_pred, data=covid, color='red')


# In[19]:


# importing the metrics from skitlearn for to get mean absolute erroe, mean squerd arroe and root mean squerd error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[21]:


#importing the metrics from skitlearn for to get rsquerd value
from sklearn import metrics
r_square = metrics.r2_score(y_test,y_pred)


# In[22]:


# print the r_square value
print(r_square)

